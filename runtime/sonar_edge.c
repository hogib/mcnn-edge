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

/* NEON path on AArch64 unless built with -DSE_NO_NEON (scalar reference).
 * Needs SE_W_ELEMS, which headers exported before the NEON kernel lack. */
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(SE_W_ELEMS) && !defined(SE_NO_NEON)
#define SE_USE_NEON 1
#include <arm_neon.h>
#else
#define SE_USE_NEON 0
#endif

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

#if SE_USE_NEON
/* NEON kernel (AArch64 Advanced SIMD, no dot-product extension needed, so it
 * runs on the Pi 4's Cortex-A72). Vectorised over output channels: for each
 * output pixel, tap and input channel, the input value (u8 widened to s16) is
 * broadcast against 16 int16 weights and accumulated into four int32x4 with
 * vmlal_laneq_s16. Every product fits s16 x s16 -> s32 exactly and integer
 * addition is associative, so the result is bit-identical to conv3x3().
 * Weights are transposed once to [tap][ci][co] int16 (static buffer). */
static int16_t wt16[SE_W_ELEMS];
static const int16_t *layer_wt[SE_N_LAYERS];

static void prepare_neon_weights(void)
{
    size_t off = 0;
    for (int i = 0; i < SE_N_LAYERS; i++) {
        const se_layer_t *L = &se_layers[i];
        int16_t *dst = wt16 + off;
        for (int co = 0; co < L->cout; co++)
            for (int t = 0; t < 9; t++)
                for (int ci = 0; ci < L->cin; ci++)
                    dst[((size_t)t * L->cin + ci) * L->cout + co] = L->w[((size_t)co * 9 + t) * L->cin + ci];
        layer_wt[i] = dst;
        off += (size_t)9 * L->cin * L->cout;
    }
}

#define MLA16(lane)                                                             \
    do {                                                                        \
        const int16_t *wq = wk + (size_t)(ci + (lane)) * cout;                 \
        const int16x8_t w0 = vld1q_s16(wq), w1 = vld1q_s16(wq + 8);            \
        a0 = vmlal_laneq_s16(a0, vget_low_s16(w0), x, lane);                   \
        a1 = vmlal_high_laneq_s16(a1, w0, x, lane);                            \
        a2 = vmlal_laneq_s16(a2, vget_low_s16(w1), x, lane);                   \
        a3 = vmlal_high_laneq_s16(a3, w1, x, lane);                            \
    } while (0)

/* Four adjacent output pixels (ox .. ox+3) of one row at once, for layers with
 * cin % 8 == 0: each pair of weight vectors loaded is used for 4 pixels, so
 * weight traffic per MAC drops 4x. 16 accumulators + 2 weight + 4 input
 * registers fit the 32 NEON registers. Same products and sums as the
 * single-pixel path, so the same result. */
#ifndef SE_NEON_1PX  /* -DSE_NEON_1PX: single-pixel kernel only, for comparison */
#define MLA4(acc0, acc1, acc2, acc3, xv, lane)                                  \
    do {                                                                        \
        acc0 = vmlal_laneq_s16(acc0, w0l, xv, lane);                            \
        acc1 = vmlal_high_laneq_s16(acc1, w0, xv, lane);                        \
        acc2 = vmlal_laneq_s16(acc2, w1l, xv, lane);                            \
        acc3 = vmlal_high_laneq_s16(acc3, w1, xv, lane);                        \
    } while (0)
#define LANE4(lane)                                                             \
    do {                                                                        \
        const int16_t *wq = wk + (size_t)(ci + (lane)) * cout;                 \
        const int16x8_t w0 = vld1q_s16(wq), w1 = vld1q_s16(wq + 8);            \
        const int16x4_t w0l = vget_low_s16(w0), w1l = vget_low_s16(w1);        \
        MLA4(p0a0, p0a1, p0a2, p0a3, x0, lane);                                 \
        MLA4(p1a0, p1a1, p1a2, p1a3, x1, lane);                                 \
        MLA4(p2a0, p2a1, p2a2, p2a3, x2, lane);                                 \
        MLA4(p3a0, p3a1, p3a2, p3a3, x3, lane);                                 \
    } while (0)
#define LOADX(p) vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ip + (size_t)(p) * s * cin + ci)))
#define STORE4(p, a0_, a1_, a2_, a3_)                                           \
    do {                                                                        \
        vst1q_s32(acc, a0_); vst1q_s32(acc + 4, a1_);                           \
        vst1q_s32(acc + 8, a2_); vst1q_s32(acc + 12, a3_);                      \
        uint8_t *o = op + (size_t)(p) * cout + cb;                              \
        for (int k = 0; k < 16; k++) o[k] = requant(acc[k], L->mult[cb + k], L->shift[cb + k]); \
    } while (0)

static void neon_block4(const uint8_t *in, int wp, const se_layer_t *L, const int16_t *wt,
                        int oy, int ox, uint8_t *op)
{
    const int cin = L->cin, cout = L->cout, s = L->stride, d = L->dilation;
    int32_t acc[16];
    for (int cb = 0; cb < cout; cb += 16) {
        const int32x4_t b0 = vld1q_s32(L->bias + cb), b1 = vld1q_s32(L->bias + cb + 4);
        const int32x4_t b2 = vld1q_s32(L->bias + cb + 8), b3 = vld1q_s32(L->bias + cb + 12);
        int32x4_t p0a0 = b0, p0a1 = b1, p0a2 = b2, p0a3 = b3;
        int32x4_t p1a0 = b0, p1a1 = b1, p1a2 = b2, p1a3 = b3;
        int32x4_t p2a0 = b0, p2a1 = b1, p2a2 = b2, p2a3 = b3;
        int32x4_t p3a0 = b0, p3a1 = b1, p3a2 = b2, p3a3 = b3;
        for (int ky = 0; ky < 3; ky++) {
            const int iy = oy * s + ky * d;
            for (int kx = 0; kx < 3; kx++) {
                const uint8_t *ip = in + ((size_t)iy * wp + ox * s + kx * d) * cin;
                const int16_t *wk = wt + (size_t)(ky * 3 + kx) * cin * cout + cb;
                for (int ci = 0; ci < cin; ci += 8) {
                    const int16x8_t x0 = LOADX(0), x1 = LOADX(1), x2 = LOADX(2), x3 = LOADX(3);
                    LANE4(0); LANE4(1); LANE4(2); LANE4(3);
                    LANE4(4); LANE4(5); LANE4(6); LANE4(7);
                }
            }
        }
        STORE4(0, p0a0, p0a1, p0a2, p0a3);
        STORE4(1, p1a0, p1a1, p1a2, p1a3);
        STORE4(2, p2a0, p2a1, p2a2, p2a3);
        STORE4(3, p3a0, p3a1, p3a2, p3a3);
    }
}
#undef MLA4
#undef LANE4
#undef LOADX
#undef STORE4
#endif /* SE_NEON_1PX */

static void conv3x3_neon(const uint8_t *in, int h, int w, const se_layer_t *L, const int16_t *wt,
                         uint8_t *out, int pad_out, int *oh, int *ow)
{
    const int cin = L->cin, cout = L->cout, s = L->stride, d = L->dilation;
    const int wp = w + 2 * d;
    const int ho = (h - 1) / s + 1, wo = (w - 1) / s + 1;
    const int wop = wo + 2 * pad_out;
    const int cin8 = cin & ~7;
    int32_t acc[16];

    for (int oy = 0; oy < ho; oy++) {
        for (int ox = 0; ox < wo; ox++) {
            uint8_t *op = out + ((size_t)(oy + pad_out) * wop + (ox + pad_out)) * cout;
#ifndef SE_NEON_1PX
            if (cin8 == cin && ox + 4 <= wo) {   /* 4-pixel block, then skip the 3 extra pixels */
                neon_block4(in, wp, L, wt, oy, ox, op);
                ox += 3;
                continue;
            }
#endif
            for (int cb = 0; cb < cout; cb += 16) {
                int32x4_t a0 = vld1q_s32(L->bias + cb), a1 = vld1q_s32(L->bias + cb + 4);
                int32x4_t a2 = vld1q_s32(L->bias + cb + 8), a3 = vld1q_s32(L->bias + cb + 12);
                for (int ky = 0; ky < 3; ky++) {
                    const int iy = oy * s + ky * d;
                    for (int kx = 0; kx < 3; kx++) {
                        const uint8_t *ip = in + ((size_t)iy * wp + ox * s + kx * d) * cin;
                        const int16_t *wk = wt + (size_t)(ky * 3 + kx) * cin * cout + cb;
                        int ci = 0;
                        for (; ci < cin8; ci += 8) {
                            const int16x8_t x = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(ip + ci)));
                            MLA16(0); MLA16(1); MLA16(2); MLA16(3);
                            MLA16(4); MLA16(5); MLA16(6); MLA16(7);
                        }
                        for (; ci < cin; ci++) {
                            const int16_t xv = ip[ci];
                            const int16_t *wq = wk + (size_t)ci * cout;
                            const int16x8_t w0 = vld1q_s16(wq), w1 = vld1q_s16(wq + 8);
                            a0 = vmlal_n_s16(a0, vget_low_s16(w0), xv);
                            a1 = vmlal_high_n_s16(a1, w0, xv);
                            a2 = vmlal_n_s16(a2, vget_low_s16(w1), xv);
                            a3 = vmlal_high_n_s16(a3, w1, xv);
                        }
                    }
                }
                vst1q_s32(acc, a0); vst1q_s32(acc + 4, a1); vst1q_s32(acc + 8, a2); vst1q_s32(acc + 12, a3);
                for (int k = 0; k < 16; k++) op[cb + k] = requant(acc[k], L->mult[cb + k], L->shift[cb + k]);
            }
        }
    }
    *oh = ho;
    *ow = wo;
}
#undef MLA16
#endif

const int32_t *se_forward(const uint8_t *frame)
{
    uint8_t *in = act_a, *out = act_b;
    int h = SE_IN_H, w = SE_IN_W;
#if SE_USE_NEON
    static int prepared = 0;
    if (!prepared) { prepare_neon_weights(); prepared = 1; }
#endif

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
#if SE_USE_NEON
        if (se_layers[i].cout % 16 == 0)
            conv3x3_neon(in, h, w, &se_layers[i], layer_wt[i], out, pad_out, &oh, &ow);
        else
#endif
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
