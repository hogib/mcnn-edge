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

### 6.1 Results: `small`, strong augmentation, 150 epochs (the shipped configuration)

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
   Section 6.3.

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

1. **Width without resolution does not help.** `small_wide` is 0.8 pooled
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
   points. The cause is not isolated: the per-layer activation scales are
   within a few percent of `small`'s and the weights are already quantised
   per channel (M2), so neither range is obviously worse. The effect is
   consistent in sign but smaller than the run-to-run noise, and would need
   a per-layer error breakdown (M2, Section 4) before it was acted on. It is
   noted because it is the first configuration for which PTQ is not free.
4. **Decision.** `small` stays the shipped configuration: `small_wide` costs
   2.6x the compute for no measurable gain. If the Pi budget allows ~100 M
   MACs, the compute should go into resolution (for instance the `base`
   layout at 480x320 with narrower channels, or `small` at 360x240), not into
   width.

## 7. Limitations

- One seed per configuration, and GPU training is not bit-reproducible: the
  measured run-to-run spread is about 4 mAP points on the development split
  (Section 5). The 5-fold protocol with pooled AP (Section 6) is the final
  evaluation; even there each fold is a single run.
- No test-time augmentation or ensembling, by design: the C runtime executes
  exactly one forward pass.
- The Wall class is evaluated like the others although its boxes are large and
  loosely defined; its AP mostly reflects annotation consistency.

## References

- Law, H. and Deng, J. (2018). CornerNet: Detecting objects as paired keypoints. ECCV.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.
- Yu, F. and Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. ICLR.
- Zhou, X., Wang, D. and Krähenbühl, P. (2019). Objects as points. arXiv:1904.07850.
