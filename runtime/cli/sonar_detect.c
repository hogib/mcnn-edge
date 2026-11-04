/* sonar-detect: run the detector on PGM frames.
 *
 *   sonar-detect [-t THR] [-n REPEAT] [-s DIR] frame.pgm [more.pgm ...]
 *
 * Prints one line per detection: "<file> <cls> <score> <x0> <y0> <x1> <y1>".
 * With -s DIR (models with a segmentation head), the per-cell label map of
 * each frame is written to DIR/<basename>.labels.pgm (P5, one byte per
 * output cell, 0 = background, c + 1 = class c).
 * With -n REPEAT > 1, each frame is run REPEAT times and the mean wall time per
 * frame is printed to stderr (benchmark mode). Input must be binary PGM (P5),
 * maxval 255, of the compiled-in size.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../sonar_edge.h"

static int read_pgm(const char *path, uint8_t *buf, int h, int w)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    char magic[3] = {0};
    int pw, ph, maxv;
    if (fscanf(f, "%2s %d %d %d", magic, &pw, &ph, &maxv) != 4 || strcmp(magic, "P5") != 0 || maxv != 255) {
        fprintf(stderr, "%s: not a binary 8-bit PGM\n", path); fclose(f); return -1;
    }
    if (pw != w || ph != h) {
        fprintf(stderr, "%s: size %dx%d, expected %dx%d\n", path, pw, ph, w, h); fclose(f); return -1;
    }
    fgetc(f); /* single whitespace after maxval */
    size_t n = fread(buf, 1, (size_t)h * w, f);
    fclose(f);
    return n == (size_t)h * w ? 0 : -1;
}

int main(int argc, char **argv)
{
    float thr = -4.0f;
    int repeat = 1, i = 1;
    const char *segdir = NULL;
    for (; i < argc && argv[i][0] == '-'; i++) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) thr = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) segdir = argv[++i];
        else { fprintf(stderr, "usage: %s [-t THR] [-n REPEAT] [-s DIR] frame.pgm ...\n", argv[0]); return 2; }
    }
    const int h = se_input_height(), w = se_input_width();
    uint8_t *frame = malloc((size_t)h * w);
    se_detection_t dets[64];
    double total_s = 0; long runs = 0;
    for (; i < argc; i++) {
        if (read_pgm(argv[i], frame, h, w)) return 1;
        int n = 0;
        for (int r = 0; r < repeat; r++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            n = se_detect(frame, thr, dets, 64);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_s += (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
            runs++;
        }
        for (int k = 0; k < n; k++)
            printf("%s %d %.6f %.3f %.3f %.3f %.3f\n", argv[i], dets[k].cls, dets[k].score,
                   dets[k].x0, dets[k].y0, dets[k].x1, dets[k].y1);
        if (segdir) {
            if (se_seg_classes() <= 0) { fprintf(stderr, "-s: the compiled-in model has no segmentation head\n"); return 1; }
            const int oh = se_seg_height(), ow = se_seg_width();
            uint8_t *labels = malloc((size_t)oh * ow);
            se_segment_head(se_forward(frame), labels);
            const char *base = strrchr(argv[i], '/');
            base = base ? base + 1 : argv[i];
            char path[4096];
            snprintf(path, sizeof path, "%s/%s.labels.pgm", segdir, base);
            FILE *f = fopen(path, "wb");
            if (!f) { perror(path); free(labels); return 1; }
            fprintf(f, "P5\n%d %d\n255\n", ow, oh);
            fwrite(labels, 1, (size_t)oh * ow, f);
            fclose(f);
            free(labels);
        }
    }
    if (repeat > 1 && runs) fprintf(stderr, "mean %.2f ms/frame over %ld runs\n", 1e3 * total_s / runs, runs);
    free(frame);
    return 0;
}
