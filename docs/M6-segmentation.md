# M6: semantic segmentation head

## 1. Motivation

The dataset ships a per-pixel label map for every frame (`Masks/`, LabelMe,
values 0 = background and 1..11 = the classes in sorted order, the same order
as `class_names`). A detector that also outputs a coarse class map costs
almost nothing on top of the box head and gives a second, pixel-level view of
the same objects. This milestone asks two questions: does a segmentation head
on the shared features change the box accuracy, and how good a mask can the
stride-8 grid of the shipped topology produce.

## 2. Method

### 2.1 Head and targets

A second 1x1 convolution of C + 1 = 12 channels on the same stride-8 features
(`Detector(seg=True)`, head channels 15 -> 27, +0.3 M MACs at 360x240, +396
parameters). The target of a cell is the label of the pixel at its centre,
taken from the label map resized to the working resolution by nearest
neighbour. Masks follow the box augmentation exactly: horizontal flip,
integer translation and the random zoom (nearest interpolation, zero fill =
background).

### 2.2 Loss

Per-cell cross-entropy with a class weight on background,

    L = L_heat + 0.1 L_size + L_off + w_seg * CE(seg, target; weight_bg),

with w_seg = 1. Background covers 95.6% of the pixels (foreground 4.4%: tire
0.8%, bottle 0.5%, the rest below 0.3% each), so with weight_bg = 1 the head
learns to predict background almost everywhere (Section 3.1); weight_bg =
0.1 is the setting used.

### 2.3 Evaluation

The per-cell argmax is upsampled by the stride (nearest) to the working
resolution and compared with the label map there; the (C+1)x(C+1) confusion
matrix gives per-class IoU and mIoU (mean over the 12 classes; "foreground"
= mean over the 11 object classes). Under 5-fold cross-validation the five
test-fold confusion matrices are summed before IoU is taken ("pooled mIoU",
`sonar-edge cv`), for the same reason as pooled AP in M1 §6. Model selection
stays on validation box mAP, so the boxes are evaluated exactly as in M1.

Two reference points: the mIoU of the *cell-centre labels themselves*,
upsampled the same way, is the ceiling of what this head can represent; and
bilinear instead of nearest upsampling of the logits was tested as a free
refinement.

## 3. Results

### 3.1 Development split (fold 0 test, fold 1 val; `small`, 360x240, strong aug, 150 epochs)

| Run | Box mAP (test) | mIoU | mIoU, objects | Recall / precision, bottle | Note |
|---|---|---|---|---|---|
| Boxes only, seeds 0/1/2 (M1 §6.5) | 64.0 / 61.0 / 58.0 | - | - | - | reference |
| + seg head, weight_bg = 1 (`runs/small_360_seg_e150`) | 63.3 | 27.4 | 21.0 | 32% / 86% | recall collapse |
| + seg head, weight_bg = 0.1 (`runs/small_360_segbg_e150`) | **67.3** | **35.4** | **29.8** | 76% / 46% | used below |
| Cell-centre labels (ceiling of the stride-8 grid) | - | 77.2 | 75.2 | - | representation limit |

With weight_bg = 1 every object class has high precision (86-98% for
bottle, tire, propeller, drink-carton) and recall of 10-32%: the head calls
object pixels background. Subtracting a constant from the background logit,
tuned on the validation fold (best offset 1), recovers only 2 points (test
29.6). Bilinear upsampling of the logits is 1 point *worse* than nearest
(26.4), since it blurs the boundaries of 4-cell objects. Down-weighting
background in the loss raises recall to 50-90% at precision 30-70% and
mIoU by 8 points; the boxes do not suffer (67.3, the best of the four
360x240 runs on this split, inside the seed spread).

### 3.2 5-fold cross-validation (`runs/small_360_seg_cv`, weight_bg = 0.1, seed 0)

| | Boxes, pooled mAP float / int8 | Per fold (float) | mIoU pooled | mIoU objects | mIoU per fold |
|---|---|---|---|---|---|
| 360x240 + seg | 52.1 / 52.2 | 61.7, 67.6, 48.2, 48.7, 66.3 | **33.9** (int8 33.8) | **28.2** (int8 28.1) | 34.0, 39.9, 29.8, 24.2, 35.1 |
| 360x240, boxes only, 3 seeds (M1 §6.5) | 52.6 +- 1.9 / 52.3 | 61.0, 65.1, 52.2, 48.2, 67.5 (means) | - | - | - |

Pooled per-class IoU, recall and precision (test folds summed):

| Class | IoU | Recall | Precision |
|---|---|---|---|
| Background | 96.9 | 97 | 99 |
| Bottle | 32.7 | 72 | 37 |
| Can | 27.0 | 65 | 32 |
| Chain | 32.2 | 61 | 41 |
| Drink-carton | 27.6 | 64 | 33 |
| Hook | 22.7 | 49 | 30 |
| Propeller | 23.7 | 47 | 33 |
| Shampoo-bottle | 24.6 | 71 | 27 |
| Standing-bottle | 3.2 | 6 | 7 |
| Tire | 46.1 | 75 | 55 |
| Valve | 5.4 | 10 | 11 |
| Wall | 64.5 | 91 | 69 |

Box AP per class is within the seed spread of the box-only model for every
class (bottle 86.0, tire 89.7, drink-carton 76.8, can 17.3, chain 39.0,
valve 13.5, standing-bottle 3.6). The int8 masks (per-cell argmax of the
quantised head, `sonar-edge quantize` after this milestone) pool to 33.8
mIoU against 33.9 float, per fold within 0.2, and every per-class IoU within
0.3: quantisation is free for the masks as for the boxes.

### 3.3 Sub-pixel head: a stride-4 label grid (`--seg-up 2`, `runs/small_360_seg4_cv`)

The runtime has no upsampling layer, so the finer grid is produced by the
head itself: (C+1) x 4 = 48 segmentation channels per stride-8 cell, read as
the four 2x2 sub-cells (`pixel_shuffle`, channel c*4 + sy*2 + sx for label c
of sub-cell (sy, sx)), giving a 90x60 label map at 360x240. The target is
the label at each sub-cell's centre; everything else (loss, weight_bg = 0.1,
evaluation) is unchanged. Cost: +1.6 M MACs over the stride-8 head (80.5 M
total), +1,188 parameters. The C runtime, export and golden test handle the
layout (`SE_SEG_UP`); the golden test matches all 3 x 5,400 cells.

| Label grid | Ceiling of the grid (cell-centre labels, dev test fold) | mIoU pooled, 5-fold | objects | per fold | Boxes, pooled float / int8 |
|---|---|---|---|---|---|
| stride 8 (45x30) | 77.2 | 33.9 (int8 33.8) | 28.2 | 34.0, 39.9, 29.8, 24.2, 35.1 | 52.1 / 52.2 |
| stride 4 (90x60) | 87.0 | **35.0** (int8 35.0) | **29.4** | 35.3, 42.7, 32.0, 23.1, 35.5 | 52.0 / 53.5 |
| stride 2 (not run) | 93.2 | - | - | - | - |

Per class, stride 4 gains bottle (+3), tire (+4), shampoo-bottle (+4),
drink-carton (+3) and propeller (+2) and loses can (-2) and standing-bottle
(-1); recall and precision move by at most a few points (tire recall 75 ->
83). Best epochs 68, 96, 91, 142, 84.

### 3.4 Dice term (`--seg-dice 1`, `runs/small_360_segdice_cv`)

A soft multi-class Dice loss, 1 - (2 |p ∩ t| + 1) / (|p| + |t| + 1) per
class on the softmax probabilities, averaged over the classes present in
the batch (background included), is added with weight 1 to the weighted
cross-entropy of Section 3.3; nothing else changes (stride-4 head,
weight_bg = 0.1, seed 0).

| | stride-8 head (3.2) | stride-4 head (3.3) | stride-4 + Dice |
|---|---|---|---|
| mIoU pooled (int8) | 33.9 (33.8) | 35.0 (35.0) | **41.2** (41.3) |
| mIoU, object classes | 28.2 | 29.4 | **36.1** |
| mIoU per fold | 34.0, 39.9, 29.8, 24.2, 35.1 | 35.3, 42.7, 32.0, 23.1, 35.5 | 40.0, 49.0, 39.0, 31.0, 40.4 |
| Boxes, pooled mAP float (int8) | 52.1 (52.2) | 52.0 (53.5) | **49.8** (49.0) |
| Boxes per fold | 61.7, 67.6, 48.2, 48.7, 66.3 | 67.9, 61.4, 49.7, 47.1, 68.2 | 57.5, 66.0, 46.2, 47.4, 66.5 |
| Box-only reference, 3 seeds | | | 52.6 +- 1.9; per fold 61.0, 65.1, 52.2, 48.2, 67.5 |

Per-class IoU with Dice: bottle 45.4 (+9), drink-carton 44.3 (+14), tire
56.2 (+6), hook 37.9 (+15), propeller 31.9 (+6), can 33.2 (+8), chain 34.5
(+2), wall 67.5 (+4); standing-bottle 7.1 and valve 8.7 stay low. The gain
is almost entirely precision: bottle 42 -> 54, drink-carton 38 -> 64, tire
56 -> 74, hook 27 -> 48, propeller 37 -> 50, at recall within a few points
of before (bottle 72 -> 75, tire 83 -> 70, drink-carton 61 -> 59). The halo
of Section 4, item 2 is what the Dice term removes. Best epochs 60, 95, 78,
113, 104.

The boxes lose 2.2 pooled points against the stride-4 run and 2.8 against
the three-seed box-only mean, i.e. about 1.5 seed standard deviations, with
per-fold differences to the box-only means of -3.5, +0.9, -6.0, -0.8, -1.0.
The per-class box losses are in chain (-7), tire (-4), shampoo-bottle (-7)
and valve (-14), the last two among the seed-noisy classes; can (+4) and
drink-carton (+1) gain. One seed, so this is suggestive rather than
established, but it is the first segmentation setting whose box mAP is
below every seed of the box-only model.

## 4. Discussion

1. **The boxes are unaffected.** Pooled 52.1 against 52.6 +- 1.9 for the
   box-only model over three seeds, and every per-class AP inside the seed
   spread. Sharing the backbone with a dense target neither helps nor hurts
   the detector at this capacity.
2. **The masks are limited by learning, not by the grid.** The stride-8
   cell representation can reach 77 mIoU and the stride-4 grid 87; the head
   reaches 34 and 35. Halving the stride (Section 3.3) is worth +1.1 pooled
   mIoU, positive on four folds of five, at +1.6 M MACs, and the boxes do
   not move. Recall is 50-80% for the common classes at precision 30-55%,
   i.e. the head finds most object pixels but also labels background around
   them, and the finer grid does not change that pattern, so the halo is a
   property of the features and the loss rather than of the output
   resolution. The Dice term (Section 3.4) confirms this: it raises
   precision by 10-25 points per class and mIoU by 6.2 pooled, on every
   fold, with the same grid and features.
5. **Multi-task trade-off.** With cross-entropy only, the boxes were
   unaffected by the segmentation head (items 1 and 2). With the Dice term
   the masks improve and the boxes lose about 3 pooled points (Section
   3.4): the shared 43 k-parameter backbone is now pulled towards the
   dense objective. The two ways out are a smaller Dice weight (0.3 rather
   than 1) and a wider head or backbone; which one restores the boxes
   without giving back the masks is the next experiment. Until then the
   box-only model stays shipped, and the Dice model is the segmentation
   reference.
3. **Few-scene classes fail as they do for boxes.** Standing-bottle (3.2)
   and valve (5.4) are the classes concentrated in one scene (M0 §3.3);
   the segmentation head cannot learn them from the other folds either.
4. **Cost and deployment.** +0.3 M MACs and +396 parameters. The C runtime
   computes the 27-channel head map and, since 4 November, the per-cell
   label map: `se_segment` / `se_segment_head` (M3 §2.5) take the argmax on
   the real-valued channels in the Python operation order, and the golden
   test checks the labels against the Python int8 model cell by cell (3 x
   1350 cells, all equal, on `runs/small_360_seg_cv_f0`). `sonar-detect -s
   DIR` writes one PGM label map per frame. The shipped `weights.h` is the
   box-only 360x240 model (`SE_SEG_COUT` 0); exporting a `--seg` run instead
   enables the functions with no other change; `--seg-up 2` models export
   with `SE_SEG_UP 2` and a 90x60 label map.

## 5. Limitations

- One seed for the segmentation CV; the box-only comparison has three.
- The C label map is verified on the three golden frames and by the int8
  mIoU of the Python reference; the C binary has not been scored for mIoU
  over a whole fold (c_eval scores boxes only).
- mIoU is measured at the working resolution (360x240) against the
  nearest-resized label map, not at the original 480x320.
- weight_bg = 0.1 was chosen from a single comparison with 1.0, not swept.
- The stride-4 result is one seed; its +1.1 is below the ~2-point pooled seed
  noise of M1 §6.5, although the sign is consistent over four folds. The
  Dice result is also one seed; its +6.2 mIoU is well outside that noise,
  its -2.8 box mAP is at about 1.5 standard deviations and needs a seed
  repeat before it is acted on.
- The per-cell target is the centre pixel's label; objects thinner than a
  cell can be missed by the target itself (part of the 77-point ceiling).
