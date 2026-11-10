# M1: a single-stage anchor-free detector for whole sonar frames

## 1. Motivation

The predecessor study (`../sonar_cnn`) showed that classification accuracy on
hand-cropped sonar patches is dominated by two artefacts: the crop size, which
alone predicts the class with 98% accuracy, and near-duplicate frames shared
between random train/test splits. A deployable system must instead take the
whole frame and localise objects itself. We therefore formulate the task as
detection on the 320x480 frame, with a single fully convolutional network that
emits both localisation and class, so that (i) there is exactly one graph to
port to the C runtime (M3), (ii) all tensor shapes are fixed at compile time,
and (iii) object size is a regressed output rather than an input cue.

## 2. Method

### 2.1 Network

The network is a plain stack of 3x3 convolutions, each followed by batch
normalisation and ReLU, with no residual connections, upsampling or attention,
so the runtime needs a single convolution primitive. Three stride-2 layers bring
the input to stride 8; the remaining layers use dilations 1, 2, 4, 8 to grow the
receptive field without further downsampling (Yu and Koltun, 2016). A 1x1
convolution produces, per output cell, C class logits, a box size (w, h) in
stride units and a centre offset (dx, dy) in [0, 1).

| Config | Input | Layers (out channels, stride, dilation) | Params | MACs / frame | RF |
|---|---|---|---|---|---|
| base | 480x320 | (16,2,1) (32,2,1) (48,2,1) (48,1,1) (48,1,2) (48,1,4) (48,1,8) | 102,831 | 283.7 M | 255 px |
| wide | 480x320 | as base with 64 channels from layer 3 | 172,351 | 450.2 M | 255 px |
| small | 240x160 (2x downsampled) | (16,2,1) (32,2,1) (32,2,1) (32,1,1) (32,1,2) (32,1,4) | 42,463 | 34.8 M | 254 px (at full res) |

The receptive field of 255 px covers the largest objects (chains, median 121x149
px, up to 253 px). The output grid is 60x40 for `base`/`wide` and 30x20 for
`small`; the smallest objects (drink cartons, median 27x30 px) span 3-4 cells in
the former and 1.5-2 in the latter.

### 2.2 Targets and losses

We follow CenterNet (Zhou et al., 2019). For each ground-truth box of class c
with centre p = (x, y) in input pixels, let p~ = p / S be the centre on the
output grid of stride S and p^ = floor(p~) its cell. A Gaussian

    Y_{c, q} = exp( -|q - p^|^2 / (2 sigma^2) ),   sigma = (2r + 1) / 6

is splatted (elementwise maximum) onto the class-c heatmap, with the radius r
chosen as in CornerNet (Law and Deng, 2018) so that a box displaced by r still
overlaps the original at IoU >= 0.7. The size target is (w, h) / S and the
offset target p~ - p^.

The heatmap is trained with the penalty-reduced pixelwise focal loss
(Lin et al., 2017; Zhou et al., 2019),

    L_k = -(1/N) sum_{c,q} { (1 - P)^2 log P            if Y = 1
                            { (1 - Y)^4 P^2 log(1 - P)  otherwise,

where P = sigmoid of the logit and N is the number of objects, and the size and
offset heads with an L1 loss evaluated at object centres only. The total loss is
L = L_k + 0.1 L_size + L_off (CenterNet's weights).

### 2.3 Decoding

Peaks are the cells where the logit equals its 3x3 maximum; the top-K (K = 32)
peaks above a threshold are kept, and boxes are reconstructed from the size and
offset at the peak. Because the sigmoid is monotone, suppression and
thresholding are done on logits: the C decoder never evaluates a sigmoid.

### 2.4 Training protocol

AdamW (weight decay 1e-4), one-cycle schedule (peak learning rate 2e-3, 10%
warm-up), batch size 16, 60 epochs, seed 0. Augmentation ("basic"): horizontal
flip, integer translation of up to 8 px, multiplicative gain in [0.7, 1.3] and
additive bias in [-0.1, 0.1]. The model is selected on validation mAP (best
epoch); the test fold is evaluated once with that model.

Evaluation follows the scene-fold protocol of M0: development results use fold 0
as test and fold 1 as validation (1,041 / 388 / 439 frames). The metric is
VOC-style AP at IoU 0.5 with all-point interpolation, computed per class and
averaged over the classes present in the split (own implementation in
`metrics.py`, reused unchanged to score the C runtime in M4).

## 3. Results (development split, 60 epochs)

| Config | Best epoch | Val mAP@0.5 | Test mAP@0.5 |
|---|---|---|---|
| base | 46 | 59.4 | 53.7 |
| small | 48 | 54.1 | 58.6 |

Per-class AP on the test fold (fold 0); the number of test boxes is given
because several classes are small:

| Class (test boxes) | base | small |
|---|---|---|
| Bottle (201) | 80 | 72 |
| Can (65) | 20 | 24 |
| Chain (46) | 42 | 92 |
| Drink-carton (83) | 32 | 51 |
| Hook (39) | 76 | 79 |
| Propeller (64) | 56 | 30 |
| Shampoo-bottle (8) | 22 | 25 |
| Standing-bottle (2) | 100 | 100 |
| Tire (210) | 93 | 89 |
| Valve (4) | 2 | 7 |
| Wall (182) | 69 | 76 |

## 4. Discussion

1. **Compute is not the bottleneck at this scale.** `small` runs on a frame
   downsampled 2x with 8x fewer multiply-accumulates and is not worse than
   `base`. The two differ class by class (chain, drink-carton favour `small`;
   propeller favours `base`) by amounts that exceed what one would expect from
   the architecture alone, which points to run-to-run variance on a small test
   fold rather than to a systematic effect.
2. **Overfitting.** For both configurations validation mAP plateaus after
   roughly 35 epochs while the training losses continue to fall. With 1,041
   training frames from a single tank this is expected; stronger augmentation
   is the obvious remedy and is evaluated in Section 5.
3. **Weak classes.** Cans (median 40x40 px), drink cartons (27x30) and shampoo
   bottles (32x99, low contrast) are the weakest classes with adequate support.
   Valve and standing-bottle have 4 and 2 test boxes in fold 0 and their AP is
   not interpretable on this split; only the 5-fold protocol can evaluate them.
4. **Honest numbers are far from the paper's.** The MCNN paper reports 99.87%
   classification accuracy on hand crops. Detecting and classifying the same
   objects in whole frames, with no scene shared between training and test,
   gives roughly 55-60 mAP@0.5. The gap is the price of removing both leaks.

## 5. Iteration: schedule, augmentation, width

Four further runs on the same split test, one factor at a time, the three
remedies suggested by Section 4: a longer schedule, stronger augmentation and a
wider network. "Strong" augmentation adds to the basic set a random zoom
(factor 0.75-1.3 about a random point, zero fill, boxes scaled and clipped and
dropped when less than 30% of their area or 4 px remain; applied with
probability 0.7) and additive Gaussian noise with a standard deviation drawn
uniformly from [0, 0.04]. The `wide` configuration keeps the `base` topology
with 1.5x the channels at full resolution. All runs use 150 epochs with the
same one-cycle schedule stretched accordingly; the last row repeats the
`small` run with an identical seed and is discussed below.

| Run | Config | Input | MACs | Aug | Best epoch | Val mAP | Test mAP |
|---|---|---|---|---|---|---|---|
| Sec. 3 | base | 480x320 | 284 M | basic, 60 ep | 46 | 59.4 | 53.7 |
| Sec. 3 | small | 240x160 | 35 M | basic, 60 ep | 48 | 54.1 | 58.6 |
| schedule | base | 480x320 | 284 M | basic, 150 ep | 53 | 57.8 | 58.5 |
| + aug | base | 480x320 | 284 M | strong, 150 ep | 88 | 66.1 | 63.1 |
| + aug | small | 240x160 | 35 M | strong, 150 ep | 94 | 55.0 | **64.1** |
| + aug | wide | 480x320 | 450 M | strong, 150 ep | 72 | 67.8 | **68.8** |
| repeat | small | 240x160 | 35 M | strong, 150 ep | 81 | 59.9 | 60.3 |

Per-class AP on the test fold for the strongly augmented runs:

| Class (test boxes) | base | small | wide | small, repeat |
|---|---|---|---|---|
| Bottle (201) | 95 | 93 | 97 | 83 |
| Can (65) | 24 | 11 | 36 | 14 |
| Chain (46) | 47 | 86 | 48 | 36 |
| Drink-carton (83) | 55 | 34 | 73 | 53 |
| Hook (39) | 95 | 82 | 82 | 84 |
| Propeller (64) | 62 | 58 | 61 | 58 |
| Shampoo-bottle (8) | 22 | 26 | 26 | 60 |
| Standing-bottle (2) | 100 | 100 | 100 | 100 |
| Tire (210) | 94 | 89 | 99 | 93 |
| Valve (4) | 21 | 50 | 51 | 5 |
| Wall (182) | 78 | 77 | 84 | 76 |

**Findings.**

1. **Strong augmentation is the effective change.** With the schedule held at
   150 epochs it raises `base` from 58.5 to 63.1 test mAP (57.8 to 66.1 on
   validation), and the best epoch moves from the 50s to the 70s-90s: the
   network keeps improving instead of overfitting after 35 epochs as in
   Section 4. The longer schedule on its own changes little (53.7 to 58.5 on
   test, 59.4 to 57.8 on validation; opposite signs, i.e. noise).
2. **Run-to-run noise is about 4 mAP points on this split.** The repeated
   `small` run differs from the original only by the non-deterministic order of
   GPU reductions (same seed, data order and augmentation stream) and lands
   3.8 points lower on test and 4.9 higher on validation. Per-class AP swings
   by up to 50 points for classes with fewer than 50 test boxes (chain,
   shampoo-bottle, valve). Differences between single runs below this size are
   therefore not interpretable, and the only valid comparison between
   configurations is the pooled 5-fold protocol (Section 6).
3. **Width helps on this split, resolution does not.** `wide` leads `base` by
   5.7 points on test and 1.7 on validation at 1.6x the compute; the gain is
   concentrated in can and drink-carton, the small low-contrast classes. `small`,
   at a quarter of the resolution and an eighth of the compute, is within noise
   of `base` on test (64.1 and 60.3 against 63.1) while trailing it on
   validation (55.0 and 59.9 against 66.1). The two folds rank the models
   differently, which again points to fold composition rather than to the
   architecture; the bottle and tire classes, which dominate the test fold,
   are solved at any resolution.
4. **Shipped configuration.** `small` with strong augmentation is carried
   forward to M2-M5 (`runtime/weights.h` is generated from it): it meets the
   compute budget of the plan (35 M MACs, 10 ms per frame on the laptop in
   scalar C, Section M3) with a 13x margin over `wide`, and the dev-split
   evidence does not separate it from `base`. `wide` remains the accuracy
   reference; whether its lead survives the pooled protocol is settled in
   Section 6.

## 6. Final evaluation: 5-fold cross-validation

The final numbers use the K = 5 scene-fold protocol of M0: for k = 0..4 a model
is trained with fold k as test set, fold (k+1) mod 5 as validation set (model
selection) and the other three folds as training set. Every scene is tested
exactly once, so the five test-fold prediction sets partition the dataset and
are scored jointly as one detection problem ("pooled AP", the VOC convention,
`cv.py`). Pooling is the only estimate that is defined for classes concentrated
in a few scenes (standing-bottle, valve) and it weights every box equally; the
mean and standard deviation of the per-fold mAP are reported alongside as a
measure of fold-to-fold variation. The float model and its post-training int8
quantisation (M2) are both pooled, since the C runtime reproduces the latter
exactly (M4).

### 6.1 Results: `small`, strong augmentation, 150 epochs (shipped until 4 November; see 6.4)

Pooled AP over all 1,868 frames (every scene tested once), float model and its
int8 quantisation; the number of boxes is the support of each class in the
whole dataset.

| Class (boxes) | Float AP | Int8 AP |
|---|---|---|
| Bottle (468) | 78.0 | 77.4 |
| Can (292) | 9.1 | 9.6 |
| Chain (320) | 32.5 | 32.3 |
| Drink-carton (347) | 62.4 | 64.1 |
| Hook (171) | 90.8 | 91.4 |
| Propeller (197) | 51.2 | 50.9 |
| Shampoo-bottle (111) | 27.7 | 25.5 |
| Standing-bottle (65) | 1.2 | 2.5 |
| Tire (615) | 86.8 | 86.8 |
| Valve (240) | 22.8 | 23.2 |
| Wall (1056) | 81.3 | 83.4 |
| **mAP, pooled** | **49.4** | **49.7** |
| mAP, mean over folds (s.d.) | 56.4 (8.9) | 55.5 (8.5) |

Per-fold test mAP (fold 0 to 4; best epoch 81, 115, 61, 74, 102): float 60.3,
59.3, 49.9, 44.9, 67.4; int8 58.9, 56.4, 50.3, 44.7, 67.0.

**Discussion.**

1. **The pooled number is 49.4 mAP, not the 64 of the development split.** The
   development test fold (fold 0) is the easiest of the five together with
   fold 4; the folds span 45-67 mAP (s.d. 8.9), which is the fold-composition
   effect anticipated in M0 and in Section 5. The pooled estimate is lower
   than the fold mean for two reasons: it includes the classes that are absent
   or nearly absent from individual folds (standing-bottle at 1.2 AP, valve at
   22.8), which the per-fold means skip or barely weight, and it ranks
   detections of five independently trained models on one score scale, which
   penalises folds whose scores are less well calibrated.
2. **Classes concentrated in one scene are barely learnable at this scale.**
   Standing-bottle (94% of its boxes in one scene, M0) reaches 1.2 AP: when its
   scene is held out the training set contains four examples. The number is
   reported rather than excluded; it costs about 4 mAP points relative to the
   ten remaining classes. Section 6.2 shows that the full-resolution model
   does extract something from those four examples.
3. **Can is the weakest learnable class** (9.1 AP, 292 boxes, median 40x40 px,
   i.e. 20x20 px at the `small` input scale). Chain (32.5) and shampoo-bottle
   (27.7) follow. These three classes are where resolution or width would have
   to pay for itself; the dev-split result for `wide` (Section 5) suggests it
   does for can and drink-carton.
4. **Quantisation is neutral at the pooled level** (+0.3 mAP, per-class within
   +-2.2 points, no consistent sign), confirming the M2 conclusion on the whole
   dataset. Since the C runtime reproduces the int8 model exactly (M4), 49.7
   pooled mAP is the accuracy of the deployed system.

### 6.2 Results: `wide`, strong augmentation, 150 epochs (accuracy reference)

Same protocol, same folds, same seed; `wide` runs at full resolution (480x320,
450 M MACs, 13x the compute of `small`). The `small` columns are repeated from
Section 6.1 for comparison.

| Class (boxes) | wide float | wide int8 | small float | small int8 |
|---|---|---|---|---|
| Bottle (468) | 84.7 | 82.7 | 78.0 | 77.4 |
| Can (292) | 18.8 | 20.5 | 9.1 | 9.6 |
| Chain (320) | 29.8 | 30.2 | 32.5 | 32.3 |
| Drink-carton (347) | 81.4 | 79.8 | 62.4 | 64.1 |
| Hook (171) | 95.5 | 94.9 | 90.8 | 91.4 |
| Propeller (197) | 46.7 | 46.9 | 51.2 | 50.9 |
| Shampoo-bottle (111) | 30.5 | 31.4 | 27.7 | 25.5 |
| Standing-bottle (65) | 24.1 | 26.7 | 1.2 | 2.5 |
| Tire (615) | 93.9 | 92.7 | 86.8 | 86.8 |
| Valve (240) | 17.5 | 27.5 | 22.8 | 23.2 |
| Wall (1056) | 84.8 | 84.9 | 81.3 | 83.4 |
| **mAP, pooled** | **55.2** | **56.2** | 49.4 | 49.7 |
| mAP, mean over folds (s.d.) | 62.8 (8.3) | 62.4 (9.1) | 56.4 (8.9) | 55.5 (8.5) |

Per-fold test mAP (fold 0 to 4; best epoch 30, 109, 76, 88, 36): float 66.9,
68.6, 55.7, 52.1, 70.5; int8 68.6, 68.6, 53.8, 51.1, 69.7.

**Discussion.**

1. **`wide` is better by 5.8 pooled mAP points (6.5 for int8), and the lead
   is consistent.** It wins on every one of the five folds, by 3.1 to 9.3
   points (median 7.2), so the difference is not the ~4-point run-to-run
   noise of Section 5 but a property of the model. The fold ranking is the
   same for both configurations (folds 0, 1 and 4 easy, 2 and 3 hard), which
   confirms that the per-fold spread measures the data, not the model.
2. **The gain is in the small and low-contrast classes.** Drink-carton
   (+19 points), can (+10), standing-bottle (+23, from chance to 24 AP),
   bottle (+7) and tire (+7) improve; chain (-3), propeller (-5) and valve
   (-5) do not. Resolution and width therefore pay for themselves on objects
   whose extent at the `small` input scale is 20 px or less (Section 4,
   item 3), and not on the larger or more distinctive classes. The
   standing-bottle result also qualifies Section 6.1, item 2: at full
   resolution the four training examples available when its scene is held
   out are enough to detect some instances, so the class is hard rather
   than impossible under this protocol.
3. **Quantisation remains neutral** (+1.0 pooled; per class within +-2
   points except valve, +10, whose 240 boxes lie in few scenes and whose
   pooled AP is accordingly unstable).
4. **Trade-off for deployment.** The 6 points cost 13x the multiply-accumulates
   (450 M against 35 M per frame): on the laptop in scalar C that is roughly
   115 ms against 10 ms per frame (extrapolated from the `base` measurement in
   M3), and on a Raspberry Pi an estimated 0.4-0.6 s against 35-50 ms. `small`
   stays the shipped configuration under the plan's budget of ~100 M MACs;
   `wide` is the accuracy reference for the report. Whether the 6 points
   come from the input resolution or from the channel width is tested in
   Sections 6.3 and 6.4.

### 6.3 Ablation: `small` resolution with `wide` channel widths (`small_wide`)

`wide` differs from `small` in two ways, the input resolution (480x320 against
240x160) and the channel width of the dilated stack (64 against 32), and
Section 6.2 cannot separate them. `small_wide` keeps the `small` layout (six
layers, half-resolution input, 30x20 output grid) and takes the `wide`
widths: 16-32-64-64-64-64 channels, 90 M MACs (2.6x `small`, a fifth of
`wide`), 135 k parameters. Same protocol, folds and seed as above.

| Class (boxes) | small float | small_wide float | wide float | small_wide int8 |
|---|---|---|---|---|
| Bottle (468) | 78.0 | 55.9 | 84.7 | 59.8 |
| Can (292) | 9.1 | 8.3 | 18.8 | 8.3 |
| Chain (320) | 32.5 | 38.3 | 29.8 | 37.7 |
| Drink-carton (347) | 62.4 | 69.2 | 81.4 | 68.6 |
| Hook (171) | 90.8 | 95.0 | 95.5 | 92.7 |
| Propeller (197) | 51.2 | 47.6 | 46.7 | 48.0 |
| Shampoo-bottle (111) | 27.7 | 42.0 | 30.5 | 32.0 |
| Standing-bottle (65) | 1.2 | 2.0 | 24.1 | 2.4 |
| Tire (615) | 86.8 | 84.0 | 93.9 | 85.7 |
| Valve (240) | 22.8 | 28.8 | 17.5 | 27.7 |
| Wall (1056) | 81.3 | 81.0 | 84.8 | 82.5 |
| **mAP, pooled** | 49.4 | **50.2** | 55.2 | **49.6** |
| mAP, mean over folds (s.d.) | 56.4 (8.9) | 58.0 (11.9) | 62.8 (8.3) | 56.4 (12.4) |

Per-fold test mAP (fold 0 to 4; best epoch 48, 101, 39, 110, 65): float 60.1,
63.3, 51.1, 42.2, 73.4; int8 58.8, 62.2, 47.5, 41.0, 72.5. Against `small`
fold by fold: -0.2, +4.0, +1.2, -2.7, +6.0; against `wide`: -6.8, -5.3,
-4.6, -9.9, +2.9.

**Discussion.**

1. **Width without resolution does not help** (but see Section 6.6 for
   width *with* resolution). `small_wide` is 0.8 pooled
   points above `small` in float and 0.1 below in int8; it wins three folds
   and loses two, all five differences inside the ~4-point run-to-run noise
   of Section 5. `wide` beats it on four folds of five by 5 to 10 points.
   The lead of `wide` is therefore a resolution effect, which is what
   Section 4, item 3 predicted: the classes `wide` gains on (can,
   standing-bottle, drink-carton, bottle) are those whose extent at the
   half-resolution input is near or below the 8 px stride, and doubling the
   channels does not recover what the 2x downsampling has removed. The
   classes that do move with width (shampoo-bottle +14, chain +6, valve +6,
   bottle -22) move in both directions and by amounts that single folds
   explain (next item), so no per-class claim survives.
2. **The pooled bottle AP is a single-fold failure, not a trend.** Per-fold
   bottle AP for `small_wide` is 36, 78, 54, 84, 84 against 83, 81, 47, 76,
   75 for `small`: the fold 0 model (best epoch 48, chosen on fold 1
   validation) ranks its bottle detections badly, and because pooling scores
   all folds' predictions on one list, one badly calibrated fold depresses
   the pooled AP of the class by far more than its share of the boxes. The
   same mechanism explains the larger fold-to-fold spread (s.d. 11.9 against
   8.9): the wider model fits the three training folds better and transfers
   less predictably. This is the price of pooled AP, and the reason the
   per-fold numbers are reported with it.
3. **Quantisation costs `small_wide` more than the others.** The int8 model
   is below float on every fold (-0.9 to -3.6, pooled -0.6), whereas `small`
   and `wide` were neutral to slightly positive. Shampoo-bottle loses 10
   points. The per-layer activation scales are within a few percent of
   `small`'s and the weights are quantised per channel (M2); the per-layer
   error breakdown in Section 6.4, item 3 shows the quantisation error of
   this model to be no larger than `small`'s. The effect is consistent in
   sign but smaller than the run-to-run noise. It is noted because it is
   the first configuration for which PTQ is not free.
4. **Decision.** `small_wide` costs 2.6x the compute for no measurable gain
   and is not shipped. The complementary ablation, the same `small` network
   at a higher input resolution, follows in Section 6.4.

### 6.4 Ablation: `small` at 360x240 (resolution without width)

The complement of Section 6.3: the unchanged `small` network (42 k
parameters) applied to the frame downsampled by 4/3 instead of 2 (box-filter
resize, 360x240 input, 45x30 output grid), 78 M MACs per frame, 2.25x `small`
and a sixth of `wide`. Same protocol, folds and seed.

| Class (boxes) | small 240x160 | small 360x240 | wide 480x320 | small 360x240, int8 |
|---|---|---|---|---|
| Bottle (468) | 78.0 | 84.2 | 84.7 | 84.4 |
| Can (292) | 9.1 | 16.1 | 18.8 | 16.8 |
| Chain (320) | 32.5 | 41.4 | 29.8 | 39.1 |
| Drink-carton (347) | 62.4 | 69.8 | 81.4 | 71.0 |
| Hook (171) | 90.8 | 93.6 | 95.5 | 93.2 |
| Propeller (197) | 51.2 | 49.1 | 46.7 | 49.2 |
| Shampoo-bottle (111) | 27.7 | 37.6 | 30.5 | 31.4 |
| Standing-bottle (65) | 1.2 | 3.7 | 24.1 | 3.9 |
| Tire (615) | 86.8 | 84.4 | 93.9 | 82.3 |
| Valve (240) | 22.8 | 38.6 | 17.5 | 37.5 |
| Wall (1056) | 81.3 | 82.6 | 84.8 | 81.9 |
| **mAP, pooled** | 49.4 | **54.6** | 55.2 | **53.7** |
| mAP, mean over folds (s.d.) | 56.4 (8.9) | 59.5 (9.3) | 62.8 (8.3) | 58.3 (9.7) |

Per-fold test mAP (fold 0 to 4; best epoch 102, 145, 72, 101, 61): float
64.0, 66.5, 52.3, 47.0, 67.7; int8 63.8, 65.4, 51.5, 44.7, 66.2. Against
`small` fold by fold: +3.7, +7.2, +2.4, +2.1, +0.3; against `wide`: -2.9,
-2.1, -3.4, -5.1, -2.8.

**Discussion.**

1. **Resolution is the effective factor.** At 360x240 the `small` network
   gains 5.2 pooled points, 90% of the 5.8-point lead of `wide`, with no
   change to the network and 2.25x the compute (against 13x). It beats
   `small` on every fold, by 0.3 to 7.2 points, and trails `wide` on every
   fold by 2 to 5 points. Together with Section 6.3 (width alone: +0.8,
   inside the noise) this settles the question of Section 6.2: the
   improvement of `wide` comes from the input resolution, and most of it is
   available at a fifth of its cost.
2. **The gains are in the same classes as for `wide`,** bottle (+6), can
   (+7), drink-carton (+7), plus chain (+9), shampoo-bottle (+10) and valve
   (+16), the last three of which `wide` did not improve. Tire (-2) and
   propeller (-2) are flat. Standing-bottle stays at chance (3.7); the 24 AP
   that `wide` reaches on it at full resolution is the one gain that 360x240
   does not reproduce, consistent with its 25 px median width being the
   smallest extent in the dataset (M0, Section 2).
3. **Quantisation costs about one point** (-0.9 pooled; every fold
   negative, -0.2 to -2.3), as for `small_wide` and unlike `small` and `wide`.
   A per-layer breakdown (`sonar_edge.quant_diag`, 48 test frames per fold)
   finds no mechanism in the arithmetic: the relative RMS error of the
   dequantised activations is 3-8% per layer for every configuration (the
   first two layers of `small` on fold 0: 3.1%, 5.6%; of the 360x240 model:
   3.0%, 5.3%), fewer than 0.02% of activations saturate, and the head
   logits at the float model's peaks differ by 0.04-0.07 on average with a
   bias of -0.01, for `small`, `small_wide` and the 360x240 model alike. The
   per-class shampoo-bottle differences that dominate the pooled loss are
   not specific either: under int8 that class swings by up to 30 AP points
   on single folds for every configuration, including `small` (fold 1: 36
   to 6) and `wide` (fold 0: 29 to 65), because its 111 boxes lie in few
   scenes. The consistent sign over five folds is therefore unexplained
   but its magnitude is at the level of score-ordering noise, and the int8
   model remains 4.0 pooled points above `small`.
4. **Resolution saturates at 360x240.** The same network at full resolution
   (480x320, 139 M MACs, `runs/small_480_cv`) pools 54.1 float / 53.9 int8,
   per fold 64.9, 65.8, 57.6, 47.0, 72.7 (best epochs 104, 129, 129, 62, 98):
   no better than 360x240 (-0.5 pooled; per fold +0.9, -0.7, +5.3, 0.0,
   +5.0, i.e. gains on the two folds where 360x240 was weakest and nothing
   elsewhere) and 1.1 pooled points below `wide`, which it beats on two
   folds. The resolution curve of the `small` network is therefore 49.4 ->
   54.6 -> 54.1 pooled mAP for 35 -> 78 -> 139 M MACs. (The next sentence
   of the original text, that the last point of `wide` is what its doubled
   channels buy at full resolution, is superseded by Section 6.6: at
   360x240 the same channels buy +6.4.)
   Per class, full resolution helps drink-carton (80.6, equal to `wide`) and
   tire (93.5) and loses chain (35.4) and valve (21.4) relative to 360x240,
   which is consistent with the 360x240 result being near the top of the
   curve rather than on its rising part.
5. **Seed repeats** are in Section 6.5: three seeds each of `small` and of
   the 360x240 model give pooled means of 48.1 +- 1.1 and 52.6 +- 1.9, a
   gain of +4.5 that holds on every fold's mean and for every pairing of
   seeds. The int8 difference changes sign between seeds (+0.3, -0.7,
   +0.3 for `small`; -0.9, +0.6, -0.6 for 360x240), which closes item 3:
   quantisation is neutral within seed noise.
6. **Deployment.** At 78 M MACs the 360x240 model is within the plan's ~100 M
   MAC budget; the C runtime takes the input geometry from the exported
   header, so the only change for deployment is to regenerate `weights.h`
   from a 360x240 run. The expected cost is about 22 ms per frame on the
   laptop (2.25x the 9.9 ms of M4) and 80-110 ms on a Pi 4 by the M3
   estimate. Going to full resolution instead (139 M MACs, 2x the cost
   again) buys nothing (item 4).
7. **Shipped on 4 November.** After the seed repeats of Section 6.5
   confirmed the gain, `runtime/weights.h` and the golden vectors were
   regenerated from `runs/small_360_cv_f0`, the seed-0 run of this
   configuration on the development split (val 59.4 / test 64.0 float,
   int8 63.8; seed 0 is the default seed used for every configuration, not
   a seed selected on the test fold). The golden test passes bit-exact on
   the 45x30x15 head map, the C binary scores 63.81 mAP on the test fold
   against 63.80 for the Python int8 model (M4), and the laptop cost is
   20.1 ms per frame (M3). The 240x160 model remains available as
   `runs/small_e150_aug` and is the fallback if the Pi latency of the
   360x240 model turns out too high.

### 6.6 Width at 360x240: the interaction

Section 6.3 found that doubling the channel width at 240x160 is worth
nothing, Section 6.4 that raising the resolution to 360x240 is worth +4.5.
The two factors were not tested together. `small_wide` at 360x240
(`runs/small_wide_360_cv`, 203.5 M MACs, 135 k parameters; same protocol,
seed 0) closes that gap:

| Configuration | MACs | Pooled float | Pooled int8 | Per fold (float) |
|---|---|---|---|---|
| small, 240x160 (3 seeds) | 35 M | 48.1 +- 1.1 | 48.1 | 55.5, 58.5, 48.9, 42.5, 67.4 (means) |
| small_wide, 240x160 | 90 M | 50.2 | 49.6 | 60.1, 63.3, 51.1, 42.2, 73.4 |
| small, 360x240 (3 seeds, shipped) | 78 M | 52.6 +- 1.9 | 52.3 | 61.0, 65.1, 52.2, 48.2, 67.5 (means) |
| small_wide, 360x240, seed 0 | 204 M | 59.0 | 58.8 | 70.4, 72.2, 59.2, 55.5, 71.3 |
| small_wide, 360x240, seed 1 | 204 M | 53.9 | 53.7 | 64.4, 65.6, 57.4, 52.0, 67.8 |
| small_wide, 360x240, seed 2 (added 2026-11-10) | 204 M | 57.8 | 57.6 | 68.2, 66.1, 57.7, 48.3, 73.6 |
| **small_wide, 360x240, mean of 3 (s.d.)** | 204 M | **56.9 (2.7)** | **56.7 (2.6)** | 67.7, 68.0, 58.1, 52.0, 70.9 (means) |
| wide, 480x320 | 450 M | 55.2 | 56.2 | 66.9, 68.6, 55.7, 52.1, 70.5 |

Best epochs 72, 114, 113, 77, 63 (seed 0), 87, 109, 111, 104, 67 (seed
1) and 129, 114, 67, 117, 102 (seed 2). Per class (pooled float, seed 0 /
seed 1 / seed 2): bottle 86.6 / 76.6 / 88.1, can 19.6 / 14.9 / 18.0, chain
34.1 / 30.4 / 35.7, drink-carton 85.6 / 83.9 / 82.5, hook 95.3 / 90.5 / 96.5,
propeller 50.8 / 53.3 / 48.7, shampoo-bottle 36.6 / 33.9 / 34.0,
standing-bottle 39.6 / 13.6 / 17.7, tire 91.9 / 95.6 / 94.6, valve 25.4 /
18.1 / 34.5, wall 83.2 / 82.2 / 85.4.

**Discussion.**

1. **Width and resolution interact, by about +4 points.** At 240x160 the
   extra channels buy +0.8 (noise); at 360x240 they buy +4.3 on three-seed
   means (56.9 +- 2.7 against 52.6 +- 1.9), every fold mean is up (by 2.9
   to 6.7), and each seed beats the `small` run of the same seed (+4.4,
   +1.7, +6.8). The third seed (57.8, run 2026-11-10) landed between the
   first two, so the two-seed estimate (56.4 +- 3.6) was not inflated by
   the seed-0 run; the spread is still 1.4x that of `small`, and seed 1
   (53.9) remains inside the `small` seed range (51.0-54.6). The
   conclusion of Section 6.3 holds as stated, width *without* resolution
   does not help; the sentence in Section 6.4, item 4, that the last point
   of `wide` "is what its doubled channels buy at full resolution" is
   superseded, since the same channels at 360x240 give 56.9 +- 2.7 against
   55.2 for `wide` at 480x320 (one seed) at 45% of its compute, i.e. at
   least as much.
2. **The seed spread is in the pooled-AP-sensitive classes.** Between the
   two seeds, standing-bottle moves from 39.6 to 13.6, bottle from 86.6 to
   76.6 and valve from 25.4 to 18.1; drink-carton (85.6 / 83.9, from 63.1
   for `small`), tire (91.9 / 95.6, from 89.3) and hook (95.3 / 90.5) are
   the stable gains. Seed 2 has standing-bottle at 17.7, close to seed 1,
   so the 39.6 of seed 0 is the outlier; valve (34.5) is the class that
   moves most in seed 2.
3. **Deployment.** 204 M MACs is 2.6x the shipped model and twice the
   plan's budget. Written when the Pi latency was estimated at 180-260 ms;
   measured in M5 (2026-11-10) at 300 ms scalar and 46.5 ms with the
   4-pixel NEON kernel (21.5 fps on one Pi 4 core), which makes it a
   deployment candidate after all (docs/M5-benchmark.md). With three seeds
   it is at least the equal of `wide` as the accuracy reference at under
   half the cost. The 48-channel interpolation is Section 6.7.

### 6.7 A 48-channel stack at 360x240

`small48` (added 2026-11-10): `small_wide` with 48 instead of 64 channels in
layers 3-6, (16,2,1) (32,2,1) (48,2,1) (48,1,1) (48,1,2) (48,1,4); 82 k
parameters, 131.6 M MACs at 360x240 (1.7x `small`, 0.65x `small_wide`). 48
is a multiple of 16, so the NEON kernel applies. Same protocol, seed 0
(`runs/small48_360_cv`); best epochs 103, 125, 89, 70, 55.

| Configuration | MACs | Pooled float | Pooled int8 | Per fold (float) |
|---|---|---|---|---|
| small, 360x240 (3 seeds) | 78 M | 52.6 +- 1.9 | 52.3 | 61.0, 65.1, 52.2, 48.2, 67.5 (means) |
| small48, 360x240, seed 0 | 132 M | 50.9 | 50.5 | 62.6, 68.9, 53.6, 42.1, 61.3 |
| small_wide, 360x240 (3 seeds) | 204 M | 56.9 +- 2.7 | 56.7 | 67.7, 68.0, 58.1, 52.0, 70.9 (means) |

Per class (pooled float): bottle 77.6, can 14.4, chain 26.9, drink-carton
70.4, hook 84.8, propeller 47.5, shampoo-bottle 35.2, standing-bottle 2.3,
tire 93.7, valve 23.9, wall 83.0.

**Discussion.** One seed, 50.9, is inside the `small` seed range (51.0-54.6
is the range of its three seeds; 50.9 is 0.9 s.d. below their mean) and
well below every `small_wide` seed. Per fold it is above the `small` mean on
folds 0-2 and below on folds 3-4 (fold 4 by 6 points). Standing-bottle
(2.3) is low, but `small` is low there too (3.7-9.4), so it accounts for at
most about 0.6 pooled points of the difference. So the
interpolation does not show a proportional gain: at one seed it is
indistinguishable from `small`, and the width gain of Section 6.6 appears
only at 64 channels. A single seed cannot separate "no gain" from "a gain
of 1-2 points hidden by seed noise"; given the cost (1.7x the shipped model
for no measurable gain) further seeds are not a priority. If an intermediate
model is wanted on the Pi, `small_wide` itself is now affordable (M5).

### 6.5 Seed repeats under the full protocol

Each configuration above is one training run per fold. To measure how much
of the differences is training noise, `small` (240x160) and `small` at
360x240 were each run three times through the complete 5-fold protocol with
seeds 0, 1, 2 (`runs/small_cv`, `small_s1_cv`, `small_s2_cv`;
`runs/small_360_cv`, `small_360_s1_cv`, `small_360_s2_cv`; aggregated by
`sonar-edge seeds`). Everything else is identical.

| | small, 240x160 | small, 360x240 |
|---|---|---|
| Pooled mAP, float, per seed | 49.4, 47.4, 47.5 | 54.6, 52.2, 51.0 |
| Pooled mAP, float, mean (s.d.) | **48.1 (1.1)** | **52.6 (1.9)** |
| Pooled mAP, int8, mean (s.d.) | 48.1 (1.6) | 52.3 (1.7) |
| Fold 0, mean (s.d.) | 55.5 (5.4) | 61.0 (3.0) |
| Fold 1 | 58.5 (0.7) | 65.1 (2.2) |
| Fold 2 | 48.9 (1.0) | 52.2 (3.0) |
| Fold 3 | 42.5 (2.7) | 48.2 (2.4) |
| Fold 4 | 67.4 (0.2) | 67.5 (0.2) |

Per-class pooled AP, mean (s.d.) over the three seeds, float:

| Class | small 240x160 | small 360x240 |
|---|---|---|
| Bottle | 73.7 (3.8) | 85.4 (1.6) |
| Can | 8.0 (1.8) | 15.3 (0.7) |
| Chain | 27.6 (8.0) | 39.6 (3.3) |
| Drink-carton | 64.7 (2.9) | 63.1 (9.7) |
| Hook | 89.0 (2.0) | 90.5 (3.5) |
| Propeller | 48.8 (2.4) | 47.3 (1.7) |
| Shampoo-bottle | 27.7 (5.2) | 33.3 (6.7) |
| Standing-bottle | 1.6 (1.0) | 6.4 (2.9) |
| Tire | 84.6 (2.9) | 89.3 (4.3) |
| Valve | 22.7 (1.1) | 26.5 (11.3) |
| Wall | 81.1 (0.3) | 82.2 (0.5) |

**Discussion.**

1. **Pooled mAP has a seed standard deviation of 1-2 points**, against 3.8
   for the single-split repeat of Section 5: pooling five independently
   trained folds averages out most of the training noise, as expected. The
   per-fold values are noisier, up to 5.4 points on fold 0 of `small`
   (seeds 60.3, 56.6, 49.6), so a single fold should not be read on its
   own. The seed-0 values reported in Sections 6.1-6.4 happen to be the
   best of their three seeds for both configurations; the means are 1.3
   and 2.0 points lower.
2. **The resolution gain survives.** 360x240 is above 240x160 by 4.5
   pooled points on the means, on every fold's mean (by 0.1 to 6.6), and
   for every one of the nine seed pairings (the worst 360x240 seed, 51.0,
   is above the best `small` seed, 49.4). The gain is concentrated in
   bottle (+12), chain (+12), can (+7), tire (+5) and shampoo-bottle (+6);
   drink-carton and propeller are flat within their spread.
3. **Which per-class numbers mean anything.** Valve (s.d. 11 at 360x240),
   drink-carton (10), chain (8), shampoo-bottle (5-7) and standing-bottle
   move by more between seeds than between configurations; their boxes lie
   in few scenes, so a change of a few detections' scores reorders the
   pooled list. Bottle, can, hook, tire and wall have s.d. below 4 and
   their differences between configurations are interpretable.
4. **Quantisation is neutral**: the int8 pooled means equal the float means
   within 0.3, and the sign of the per-seed difference varies.

## 7. Limitations

- GPU training is not bit-reproducible: the run-to-run spread is about 4
  mAP points on a single split (Section 5) and 1-2 points on the pooled
  5-fold estimate (Section 6.5). `small` and the 360x240 model have three
  seeds under the full protocol; `small_wide` at 360x240 has two (s.d.
  3.6); `wide`, `small_wide` at 240x160 and 480x320 have one, so their
  pooled values carry about +-2 points.
- No test-time augmentation or ensembling, by design: the C runtime executes
  exactly one forward pass.
- The Wall class is evaluated like the others although its boxes are large and
  loosely defined; its AP mostly reflects annotation consistency.

## References

- Law, H. and Deng, J. (2018). CornerNet: Detecting objects as paired keypoints. ECCV.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.
- Yu, F. and Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. ICLR.
- Zhou, X., Wang, D. and Krähenbühl, P. (2019). Objects as points. arXiv:1904.07850.
