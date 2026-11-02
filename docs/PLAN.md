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
| Evaluation split | By recording sequence (frame index order), never random frames | Avoids the near-duplicate leak demonstrated in sonar_cnn |

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

| | Milestone | Done when |
|---|---|---|
| M0 | Data | Frames + boxes fetched and parsed; sequence split defined and justified; box-size and class statistics; class list confirmed |
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
- Class balance along the recording may make some sequence splits miss classes;
  M0 checks this before fixing the split.
