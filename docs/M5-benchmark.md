# M5: on-device benchmark (Raspberry Pi 4) and NEON kernel

Measured 2026-11-10. Long-form version with explanations for readers outside
the project: `report/deployment.pdf`.

## 1. Setup

Raspberry Pi 4 Model B rev 1.5, 4 GB; BCM2711, 4x Cortex-A72 at 1.8 GHz
(ARMv8.0: NEON, no `sdot`/`udot`), 32 KB L1D per core, 1 MB shared L2; Debian 13,
kernel 6.18, gcc 14.2, governor `ondemand`, no fan. Runtime built natively with the
project makefile (`-std=c11 -O2 -Wall -Wextra -Werror -pedantic`); `-std=c11`
disables FP contraction, so the float box decoder rounds identically to x86.
Reference: laptop, AMD Ryzen 5 5600H, same flags.

Models: the four fold-0 CV networks `runs/small_cv_f0` (240x160, 34.8 M MACs),
`runs/small_360_cv_f0` (deployed, 78.4 M), `runs/small_wide_360_cv_f0` (203.5 M),
`runs/wide_cv_f0` (480x320, 450.2 M).

## 2. Method

- **Equivalence.** (a) golden test (`make test`, all 20,250 int32 head values of
  3 frames); (b) all 439 test-fold-0 frames through `sonar-detect` on the Pi, output
  scored on the laptop with `python -m sonar_edge.c_eval runs/<r> --pred dets.txt
  --tag pi4` (new option); (c) byte comparison of the Pi output with the laptop
  build's output on the same frames.
- **Latency.** `sonar-detect -n 50` over 10 frames spread over the fold (500
  calls of `se_detect`: network + decoder, no file I/O), 3 repeats; repeats
  agree within 1% (wide NEON: 3%), mean reported. One core.
- **Throughput.** 4 copies in parallel, frames / elapsed time.
- **Thermal.** `vcgencmd measure_temp` / `get_throttled` before and after each run.
- **Memory.** `VmHWM` from `/proc/<pid>/status` while running (the first attempt
  used `getrusage` of children from Python and reported the forked interpreter's
  11 MB; discarded). Program size from `size`.
- **Energy.** Not measured: the Pi 4 has no power sensor and no meter was
  available.

Scripts: `bench.py` (scalar, detections) and `bench2.py` (NEON, -O3, memory,
throughput) on the Pi under `~/sonar_edge_m5`; not part of the repo.

## 3. Results

### 3.1 Equivalence

Golden test passes on the Pi for all four models, scalar and NEON. Detection
output over the 439 frames is byte-identical to the laptop's (21,333 / 25,007 /
23,218 / 28,096 lines), hence identical mAP (58.94 / 63.81 / 68.12 / 68.60).
Pi-scored results in `runs/<r>/c_eval_test_pi4.json`. NEON output is byte-identical to scalar output on the full fold.

### 3.2 Latency, one core

| Model | MACs | Laptop scalar ms | Pi scalar ms (GMAC/s) | Pi NEON ms (GMAC/s) | NEON speed-up |
|---|---|---|---|---|---|
| small 240x160 | 34.8 M | 9.0 | 56.0 (0.62) | 10.2 (3.41) | 5.5x |
| **small 360x240 (deployed)** | 78.4 M | 20.1 | 125.9 (0.62) | **22.5 (3.48)** | 5.6x |
| small_wide 360x240 | 203.5 M | 27.6 | 300.1 (0.68) | 77.6 (2.62) | 3.9x |
| wide 480x320 | 450.2 M | 54.8 | 656.0 (0.69) | 140.9 (3.20) | 4.7x |

Deployed model, other builds: scalar `-O3 -mcpu=cortex-a72` 75.7 ms (gcc's
auto-vectoriser, 1.7x); NEON `-O3 -mcpu=cortex-a72` 21.5 ms (within spread of
`-O2`, so `-O2` stays). The M3 projection of 70-100 ms for the scalar Pi runtime
was too optimistic (Pi/laptop ratio 6.2 for 32-channel models, 11-12 for
64-channel ones). Both gcc 16.2 (laptop) and gcc 14.2 (Pi) auto-vectorise the
scalar dot product at -O2 (checked in the assembly), but the laptop gets 7-8
GMAC/s on 64-channel layers and the Pi 0.7; not analysed further.

### 3.3 Four cores

| Model | Kernel | Per copy ms | Frames/s | End temp |
|---|---|---|---|---|
| small 240x160 | scalar / NEON | 56.5 / 10.7-11.8 | 70.6 / 339 | 72.0 / 60.3 C |
| small 360x240 | scalar / NEON | 126.8 / 25.6-26.8 | 31.4 / 149 | 68.6 / 65.7 C |
| small_wide 360x240 | scalar / NEON | 303 / 162 | 13.2 / 24.4 | 75.9 / 71.5 C |
| wide 480x320 | scalar / NEON | 664 / 253-261 | 6.0 / 15.3 | 81.8* / 78.4** C |

No throttling in any run of the three smaller models (no fan; runs 1-3 min).
*The 4-core scalar wide run reached 81.8 C and set the sticky soft-temperature-limit
bit (`throttled=0x80000`); per-copy latency rose only 1% (664 vs 656 ms), but
sustained 4-core load needs a heat sink. **Run directly after; the flag is
cumulative, so throttling in this run cannot be ruled out. Scalar scales perfectly; NEON
32-channel models lose 5-15% per copy; 64-channel NEON loses ~50% per copy.

### 3.4 Memory and size (NEON build)

Deployed model: bss 854 KB (2 x 355 KB activations, 81 KB head map, 83 KB int16
weight copy), text 53 KB (42 KB weights), peak RSS 2.3 MB. Fallback 1.8 MB,
small_wide 2.6 MB, wide 3.4 MB.

### 3.5 Cost against MCNN

MCNN (sonar_cnn reimplementation, 305x305 crop): 2,797,706 parameters, 234.7 M
MACs per crop (counted with forward hooks); x 2.08 boxes per frame = ~490 M MACs
to classify the annotated objects of a frame, with no localisation. The deployed
detector: 78.4 M MACs for detection and classification of the whole frame, 42,463
parameters.

## 4. NEON kernel (runtime/sonar_edge.c, `conv3x3_neon`)

Vectorised over output channels: per output pixel, block of 16 output channels
(four int32x4 accumulators), tap t and input channel i, the activation x[t,i]
(u8 widened to s16, 8 at a time) multiplies 16 int16 weights with
`vmlal_laneq_s16` / `vmlal_high_laneq_s16` (4 MACs per instruction). Weights are
transposed once at first call from [co][t][ci] int8 to [t][ci][co] int16 into a
static buffer of `SE_W_ELEMS` (new define emitted by `export_c`). |w x| <=
127 * 255 < 2^15, so products are exact; int32 addition is associative and
|acc| < 2^28, so the result is bit-identical to the scalar kernel by
construction. Used on AArch64 for layers with cout % 16 == 0 (all layers here);
`-DSE_NO_NEON` forces scalar; other targets and headers without `SE_W_ELEMS`
compile the scalar path unchanged. No dot-product instructions, so it runs on
the Pi 4.

64-channel models gain less (small_wide 3.9x, wide 4.7x vs 5.6x) and small_wide
scales poorly to 4 cores. Candidate cause: the int16 weight table of a 64x64 layer
is 74 KB > 32 KB L1D, so it streams from L2 per output pixel, and four copies'
working sets exceed the 1 MB shared L2. This does not explain small_wide vs wide
(identical 64x64 layers), and cache misses were not measured: open. Lever:
process 2-4 output pixels per weight pass.

## 5. Conclusions

- Deployed model on a Pi 4: 22.5 ms / frame, 44 fps on one core, 149 fps on four,
  2.3 MB RSS; output bit-exact, so the CV accuracy holds on device. The
  240x160 fallback is not needed.
- small_wide@360 (best accuracy, 56.4 pooled) at 77.6 ms / 12.9 fps single core
  is now a deployment candidate.
- Open: energy (external USB-C meter or INA219), sustained thermal test,
  multi-pixel kernel for 64-channel layers, Pi 5 `udot` kernel.
