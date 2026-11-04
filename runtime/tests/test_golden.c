/* Bit-exact check of the C runtime against the Python integer simulation. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../sonar_edge.h"
#include "golden.h"

int main(void)
{
    const int H = se_input_height(), W = se_input_width();
    const int oh = se_out_height(), ow = se_out_width(), oc = se_head_channels();
    if (oh != GOLD_HEAD_H || ow != GOLD_HEAD_W || oc != GOLD_HEAD_C) {
        printf("FAIL: head geometry %dx%dx%d, golden %dx%dx%d\n", oh, ow, oc, GOLD_HEAD_H, GOLD_HEAD_W, GOLD_HEAD_C);
        return 1;
    }
    int failed = 0;
    for (int g = 0; g < GOLD_N; g++) {
        const uint8_t *frame = gold_images + (size_t)g * H * W;
        const int32_t *hm = se_forward(frame);
        const int32_t *ref = gold_head + (size_t)g * oh * ow * oc;
        long mism = 0, maxabs = 0;
        for (long i = 0; i < (long)oh * ow * oc; i++) {
            if (hm[i] != ref[i]) {
                mism++;
                long d = labs((long)hm[i] - (long)ref[i]);
                if (d > maxabs) maxabs = d;
            }
        }
        se_detection_t dets[32];
        int n = se_detect(frame, -4.0f, dets, 32);
        printf("frame %d: %s (%ld mismatches, max |diff| %ld), %d detections, top: cls %d score %.3f\n",
               gold_frame_index[g], mism ? "FAIL" : "ok", mism, maxabs, n, n ? dets[0].cls : -1, n ? dets[0].score : 0.f);
        failed |= mism != 0;
#if GOLD_HAS_SEG
        {
            static uint8_t labels[GOLD_SEG_H * GOLD_SEG_W];
            const long nc = (long)GOLD_SEG_H * GOLD_SEG_W;
            long segmism = 0;
            if (se_seg_height() != GOLD_SEG_H || se_seg_width() != GOLD_SEG_W || se_segment_head(hm, labels) != nc) {
                printf("FAIL: se_segment_head geometry\n"); failed = 1;
            }
            for (long i = 0; i < nc; i++) segmism += labels[i] != gold_seg[(size_t)g * nc + i];
            printf("   segmentation labels: %s (%ld of %ld cells differ)\n", segmism ? "FAIL" : "ok", segmism, nc);
            failed |= segmism != 0;
        }
#endif
    }
    printf(failed ? "GOLDEN TEST FAILED\n" : "golden test passed: C head map is bit-identical to Python for %d frames\n", GOLD_N);
    return failed;
}
