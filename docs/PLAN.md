# sonar-edge: plan

End-to-end marine-debris detection on forward-looking sonar frames, deployed as a
dependency-free C11 runtime on a Raspberry Pi. Successor to `../sonar_cnn`, which
reimplemented the MCNN paper and showed that its ~99.9% rests on two shortcuts
(crop size and near-duplicate frames). This project builds the system the paper
only gestures at, and measures what it costs.

## Decisions (agreed 2026-11-02)

| Decision | Choice | Why |
|---|---|---|
| Pipeline | One fully convolutional network on the whole 320x480 frame, anchor-free centre-heatmap head (CenterNet-style) producing per-class heatmap + box size at stride 4 | One graph to port; fixed input so memory is fully static; box size becomes a regressed output, so the crop-size leak cannot exist |
| Language | C11, no dependencies, no malloc in the inference path | Compiles everywhere; the runtime needs ~6 ops; integer-only so results are bit-identical across machines |
| Arithmetic | INT8 weights (per-channel) and activations, int32 accumulation, BatchNorm folded, TFLite-style integer requantization | No float on device; verifiable bit-exactly against a Python simulation |
| Target | Raspberry Pi (model TBD; aarch64 Linux). Laptop x86 for development | Optional NEON path behind a compile flag; plain C remains the reference |
| Training | Python (uv), in this repo under `src/sonar_edge/` | Training never runs on device; Python exports weights + golden vectors |
| Evaluation split | By scene: frames grouped by matching box layout (frame indices turned out to be shuffled), scenes assigned to 5 folds, final numbers by 5-fold cross-validation; fold 0/1 as dev test/val | Avoids the near-duplicate leak demonstrated in sonar_cnn; some classes live in a single scene, so only CV evaluates every class |

## Data

`mvaldenegro/marine-debris-fls-datasets`, folder `md_fls_dataset/data/watertank-segmentation/`:
1,868 frames (`Images/*.png`, 320x480, 1 channel) with `BoxAnnotations/*.xml`
(x y w h, 11 classes). Masks are not used. `quarry-fullsize/` (~7,000 unlabelled
real-world frames) serves only as an out-of-distribution sanity check.

## Verification chain

```
PyTorch float model
  -> Python integer simulation (numpy, exactly the C arithmetic)   [must match C bit for bit]
  -> weights.h + golden input/output vectors
  -> C runtime tests (bit-exact)
  -> C pipeline CLI mAP on test split == Python mAP
```

mAP reported in the final write-up is measured from the C binary, not PyTorch.

## Milestones

Status (2026-11-03): M0-M4 done on the development split (docs/M0-M4); M1
iteration done (strong augmentation +5 mAP; `small` at 360x240 shipped since 2026-11-04 (M1 §6.4 item 7: C 63.81 mAP dev test, 20.1 ms/frame laptop), `wide` the accuracy
reference); 5-fold CV done (M1 §6): `small` 49.4 float / 49.7 int8 pooled mAP at 240x160 (shipped until 2026-11-04), `wide` 55.2 / 56.2 at 13x the compute, `small_wide` 50.2 / 49.6 at 2.6x (width alone does not help, M1 §6.3), `small` at 360x240 52.6 +- 1.9 over 3 seeds vs 48.1 +- 1.1 for `small` (M1 §6.5; resolution is the factor and saturates there, 480x320 gives 54.1 at 4x; 360x240 now shipped; the 240x160 model is the fallback if the Pi latency is too high); segmentation head done (M6: +0.3 M MACs, pooled mIoU 33.9 / objects 28.2 with background weight 0.1, boxes unchanged at 52.1; sub-pixel stride-4 variant 35.0 at +1.6 M MACs; + Dice term 41.2 mIoU but boxes 49.8 vs 52.6 +- 1.9, one seed; C runtime emits the labels); M5 needs the Pi.

| | Milestone | Done when |
|---|---|---|
| M0 | Data | Frames + boxes fetched and parsed; scene-fold split defined and justified; box-size and class statistics; class list confirmed |
| M1 | Float reference model | mAP@0.5 and per-class AP on the sequence split; sanity run on quarry frames; compute budget (MACs) stated |
| M2 | Integer model | Post-training quantization (QAT only if it loses > 1 mAP point); bit-exact integer simulation; weights header + golden vectors exported |
| M3 | C runtime | Ops + static arena + model graph; C tests bit-exact against goldens; builds with `-std=c11 -Wall -Wextra -Werror`, no malloc |
| M4 | Pipeline CLI (Linux) | PNG frame in -> detections out; batch mode over the test split reproduces Python mAP exactly |
| M5 | Benchmark | Latency, peak RAM, binary size, energy per frame on laptop and Pi; compared with the paper's MCNN crop-classifier cost |
| M6 | Report | Extends the sonar_cnn PDF with deployment results |

## Compute budget

Target <= ~100M MACs per frame so a Pi Zero 2 class device runs several frames per
second in plain C and a Pi 4/5 runs in real time. Backbone follows VarCNN (stride-2
early convs, <= 128 channels); the frame may be downsampled 2x before the network
if M1 shows no mAP cost.

## Repo layout

```
src/sonar_edge/   Python: data, model, train, quantize, export, eval
runtime/          C11: include/, src/, tests/, weights/ (generated)
tools/            CLI and benchmark programs
platforms/        build glue per target (linux-x86, rpi)
docs/             this plan, results
data/             raw dataset (git-ignored)
```

## Risks

- 1,868 frames of one tank scene is small for a detector; small-object recall
  (hooks, end-on standing bottles) is the likely weak spot.
- Heatmap heads can be sensitive to PTQ; QAT is the fallback.
- Standing-bottle (94% of boxes in one scene), valve (66%) and shampoo-bottle
  (42%) are concentrated in single scenes: when that scene is the test fold the
  model has barely seen the class. Their per-class AP must be read with this in mind.
