/* sonar_edge runtime. See sonar_edge.h and docs/M3-runtime.md.
 *
 * Layout: activations are HWC with zero padding of `dilation` pixels on every
 * side, so a 3x3 convolution with dilation d and stride s reads, for output
 * (oy, ox) and tap (ky, kx), the padded input pixel (oy*s + ky*d, ox*s + kx*d)
 * and never needs a bounds check. Each layer writes its output already padded
 * for the next layer. Two static buffers alternate as input and output.
 */
#include "sonar_edge.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "weights.h"

static uint8_t act_a[SE_ACT_ELEMS];
static uint8_t act_b[SE_ACT_ELEMS];
static int32_t head_out[SE_OUT_H * SE_OUT_W * SE_HEAD_COUT];

int se_input_height(void) { return SE_IN_H; }
int se_input_width(void) { return SE_IN_W; }
int se_num_classes(void) { return SE_N_CLASSES; }
int se_out_height(void) { return SE_OUT_H; }
int se_out_width(void) { return SE_OUT_W; }
int se_head_channels(void) { return SE_HEAD_COUT; }
int se_seg_classes(void) { return SE_SEG_COUT > 0 ? SE_N_CLASSES + 1 : 0; }
int se_seg_up(void) { return SE_SEG_UP; }
int se_seg_height(void) { return SE_OUT_H * SE_SEG_UP; }
int se_seg_width(void) { return SE_OUT_W * SE_SEG_UP; }

/* (acc * M) >> shift, rounding half away from zero, clamped to uint8.
 * Identical to sonar_edge.quant.round_shift. */
static inline uint8_t requant(int32_t acc, int32_t M, int shift)
{
    int64_t prod = (int64_t)acc * (int64_t)M;
    int64_t half = (int64_t)1 << (shift - 1);
    int64_t r = prod >= 0 ? (prod + half) >> shift : -((-prod + half) >> shift);
    if (r < 0) return 0;
    if (r > 255) return 255;
    return (uint8_t)r;
}

/* One 3x3 layer. in: padded (h+2d)x(w+2d)xcin. out: padded for pad_out, zeroed by caller. */
static void conv3x3(const uint8_t *in, int h, int w, const se_layer_t *L, uint8_t *out, int pad_out, int *oh, int *ow)
{
    const int cin = L->cin, cout = L->cout, s = L->stride, d = L->dilation;
    const int wp = w + 2 * d;                 /* padded input width */
    const int ho = (h - 1) / s + 1, wo = (w - 1) / s + 1;
    const int wop = wo + 2 * pad_out;         /* padded output width */
    int32_t acc[256];                         /* cout <= 256 */

    for (int oy = 0; oy < ho; oy++) {
        for (int ox = 0; ox < wo; ox++) {
            for (int co = 0; co < cout; co++) acc[co] = L->bias[co];
            for (int ky = 0; ky < 3; ky++) {
                const int iy = oy * s + ky * d;
                for (int kx = 0; kx < 3; kx++) {
                    const int ix = ox * s + kx * d;
                    const uint8_t *ip = in + ((size_t)iy * wp + ix) * cin;
                    const int8_t *wp3 = L->w + ((size_t)ky * 3 + kx) * cin;  /* + co*9*cin */
                    for (int co = 0; co < cout; co++) {
                        const int8_t *wv = wp3 + (size_t)co * 9 * cin;
                        int32_t a = 0;
                        for (int ci = 0; ci < cin; ci++) a += (int32_t)wv[ci] * (int32_t)ip[ci];
                        acc[co] += a;
                    }
                }
            }
            uint8_t *op = out + ((size_t)(oy + pad_out) * wop + (ox + pad_out)) * cout;
            for (int co = 0; co < cout; co++) op[co] = requant(acc[co], L->mult[co], L->shift[co]);
        }
    }
    *oh = ho;
    *ow = wo;
}

const int32_t *se_forward(const uint8_t *frame)
{
    uint8_t *in = act_a, *out = act_b;
    int h = SE_IN_H, w = SE_IN_W;

    /* input frame -> padded buffer for layer 0 (cin = 1) */
    {
        const int d = se_layers[0].dilation;
        const int wp = w + 2 * d;
        memset(in, 0, (size_t)(h + 2 * d) * wp);
        for (int y = 0; y < h; y++) memcpy(in + (size_t)(y + d) * wp + d, frame + (size_t)y * w, (size_t)w);
    }
    for (int i = 0; i < SE_N_LAYERS; i++) {
        const int pad_out = i + 1 < SE_N_LAYERS ? se_layers[i + 1].dilation : 0;
        int oh, ow;
        memset(out, 0, sizeof(act_a));
        conv3x3(in, h, w, &se_layers[i], out, pad_out, &oh, &ow);
        h = oh;
        w = ow;
        uint8_t *t = in; in = out; out = t;
    }
    /* 1x1 head: in is now unpadded h x w x SE_HEAD_CIN */
    for (int p = 0; p < h * w; p++) {
        const uint8_t *ip = in + (size_t)p * SE_HEAD_CIN;
        int32_t *op = head_out + (size_t)p * SE_HEAD_COUT;
        for (int co = 0; co < SE_HEAD_COUT; co++) {
            const int8_t *wv = se_head_w + (size_t)co * SE_HEAD_CIN;
            int32_t a = se_head_b[co];
            for (int ci = 0; ci < SE_HEAD_CIN; ci++) a += (int32_t)wv[ci] * (int32_t)ip[ci];
            op[co] = a;
        }
    }
    return head_out;
}

/* Real-valued head value, computed as the Python reference does:
 * float32( float64(acc) * scale ). */
static inline float head_real(int32_t acc, int c)
{
    return (float)((double)acc * (double)se_head_scale[c]);
}

static inline int32_t head_at(const int32_t *hm, int y, int x, int c)
{
    return hm[((size_t)y * SE_OUT_W + x) * SE_HEAD_COUT + c];
}

/* Per-cell argmax over the segmentation channels, on the real values
 * float32(float64(acc) * scale_c) like the Python reference (torch.argmax:
 * first index on ties, hence the strict comparison). With SE_SEG_UP = r the
 * channels are laid out as torch.pixel_shuffle expects, channel
 * c*r*r + sy*r + sx for label c of sub-cell (sy, sx), and the label grid is
 * r times finer than the output grid. */
int se_segment_head(const int32_t *hm, uint8_t *labels)
{
#if SE_SEG_COUT > 0
    const int c0 = SE_N_CLASSES + 4, r = SE_SEG_UP, nl = SE_N_CLASSES + 1, lw = SE_OUT_W * r;
    for (int y = 0; y < SE_OUT_H; y++) {
        for (int x = 0; x < SE_OUT_W; x++) {
            for (int sy = 0; sy < r; sy++) {
                for (int sx = 0; sx < r; sx++) {
                    const int sub = sy * r + sx;
                    int best = 0;
                    float bv = head_real(head_at(hm, y, x, c0 + sub), c0 + sub);
                    for (int c = 1; c < nl; c++) {
                        const int ch = c0 + c * r * r + sub;
                        const float v = head_real(head_at(hm, y, x, ch), ch);
                        if (v > bv) { bv = v; best = c; }
                    }
                    labels[(size_t)(y * r + sy) * lw + (x * r + sx)] = (uint8_t)best;
                }
            }
        }
    }
    return SE_OUT_H * SE_OUT_W * r * r;
#else
    (void)hm; (void)labels;
    return -1;
#endif
}

int se_segment(const uint8_t *frame, uint8_t *labels)
{
#if SE_SEG_COUT > 0
    return se_segment_head(se_forward(frame), labels);
#else
    (void)frame; (void)labels;
    return -1;
#endif
}

int se_detect(const uint8_t *frame, float thr, se_detection_t *out, int max_out)
{
    const int32_t *hm = se_forward(frame);
    int n = 0;
    for (int c = 0; c < SE_N_CLASSES; c++) {
        for (int y = 0; y < SE_OUT_H; y++) {
            for (int x = 0; x < SE_OUT_W; x++) {
                const int32_t v = head_at(hm, y, x, c);
                /* 3x3 maximum (zero-padded like F.max_pool2d(…, 3, 1, 1); padding is -inf there,
                 * so border cells only compete with existing neighbours) */
                int is_peak = 1;
                for (int dy = -1; dy <= 1 && is_peak; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        const int yy = y + dy, xx = x + dx;
                        if (yy < 0 || yy >= SE_OUT_H || xx < 0 || xx >= SE_OUT_W) continue;
                        if (head_at(hm, yy, xx, c) > v) { is_peak = 0; break; }
                    }
                }
                if (!is_peak) continue;
                const float score = head_real(v, c);
                if (score < thr) continue;
                /* box, in the same operation order as sonar_edge.targets.decode */
                const double bw = (double)head_real(head_at(hm, y, x, SE_N_CLASSES + 0), SE_N_CLASSES + 0);
                const double bh = (double)head_real(head_at(hm, y, x, SE_N_CLASSES + 1), SE_N_CLASSES + 1);
                const double dx = (double)head_real(head_at(hm, y, x, SE_N_CLASSES + 2), SE_N_CLASSES + 2);
                const double dy = (double)head_real(head_at(hm, y, x, SE_N_CLASSES + 3), SE_N_CLASSES + 3);
                const double cx = (x + dx) * SE_STRIDE, cy = (y + dy) * SE_STRIDE;
                se_detection_t det;
                det.cls = c;
                det.score = score;
                det.x0 = (float)(cx - bw * SE_STRIDE / 2);
                det.y0 = (float)(cy - bh * SE_STRIDE / 2);
                det.x1 = (float)(cx + bw * SE_STRIDE / 2);
                det.y1 = (float)(cy + bh * SE_STRIDE / 2);
                /* insert sorted by score descending, keep the best max_out */
                int i = n < max_out ? n : max_out - 1;
                if (n < max_out) n++;
                else if (out[i].score >= det.score) continue;
                while (i > 0 && out[i - 1].score < det.score) { out[i] = out[i - 1]; i--; }
                out[i] = det;
            }
        }
    }
    return n;
}
