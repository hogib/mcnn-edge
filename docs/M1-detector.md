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

## 5. Iteration (in progress)

Four runs at 150 epochs are in progress on the same split: `base` with basic
augmentation (schedule-length control), and `base`, `small`, `wide` with strong
augmentation, which adds random zoom (factor 0.75-1.3, zero fill, boxes scaled
and clipped, dropped when less than 30% of their area or 4 px remain) and
Gaussian noise (sigma up to 0.04). Results will be appended here.

## 6. Limitations

- One seed per configuration; the small test fold makes differences of a few
  mAP points unreliable. The 5-fold protocol with pooled AP is the final
  evaluation.
- No test-time augmentation or ensembling, by design: the C runtime executes
  exactly one forward pass.
- The Wall class is evaluated like the others although its boxes are large and
  loosely defined; its AP mostly reflects annotation consistency.

## References

- Law, H. and Deng, J. (2018). CornerNet: Detecting objects as paired keypoints. ECCV.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. ICCV.
- Yu, F. and Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. ICLR.
- Zhou, X., Wang, D. and Krähenbühl, P. (2019). Objects as points. arXiv:1904.07850.
