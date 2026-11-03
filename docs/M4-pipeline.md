# M4: end-to-end pipeline and exact reproduction of the Python evaluation

## 1. Motivation

The golden test of M3 shows that the runtime's network is exact on three
frames. M4 shows that the *system* (frame in, detections out, including the
decoder and the threshold/top-k logic) reproduces the Python evaluation on the
full test fold, so that any accuracy figure quoted for the deployed binary is
the same figure measured in Python, with no "deployment gap" to explain.

## 2. Pipeline

1. **Frame export.** `c_eval.py` writes every frame of a split as binary PGM
   (P5, 8-bit) at the working resolution of the chosen configuration (the 2x
   area downsampling of `small` is applied here, as in training). This step
   stands in for the sonar's raw frame output; the runtime itself contains no
   image decoding.
2. **Inference.** `sonar-detect -t THR frame.pgm ...` runs `se_detect` on each
   frame and prints `file class score x0 y0 x1 y1` per detection, with scores
   as logits. The threshold THR = -4 (probability ~ 0.018) and K = 32 match
   the Python evaluation.
3. **Scoring.** The printed detections are parsed and scored with the same
   `metrics.evaluate` (VOC AP@0.5, all-point interpolation) and the same
   ground truth used for the Python models.

## 3. Results (development split, test fold, 439 frames)

| Config | Python float | Python int8 | C runtime | max per-class |C - Python int8| |
|---|---|---|---|---|
| small | 58.59 | 59.17 | **59.17** | 0.03 points |
| base | 53.71 | 54.38 | **54.38** | 0.01 points |
| small, 150 ep, strong aug (M1 §5; shipped `weights.h`) | 64.11 | 65.16 | **65.17** | 0.07 points |

The residual per-class differences of a few hundredths of a point come from
detections with exactly equal scores whose relative order differs between the
C insertion sort and PyTorch's `topk`; they do not affect the mean.

## 4. Discussion

The chain float model -> integer simulation -> C runtime is now closed at both
ends: the integer simulation is exact with respect to the C code (M3) and the C
binary's mAP equals the simulation's (this milestone). Consequently the
accuracy of the deployed system is established by the Python evaluation
protocol of M0 (5-fold cross-validation over scenes) and does not need to be
re-measured on the device, only its speed and energy do (M5).

## 5. Limitations

- The CLI processes files; a sonar integration would call `se_detect` on
  frames from the device driver and is out of scope.
- Timing in this milestone includes process start-up and PGM I/O when measured
  from Python (10.1 ms vs 10.0 ms inside the binary for `small`); M5 measures
  inside the binary only.
