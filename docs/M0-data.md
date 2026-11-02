# M0: data, scene reconstruction and evaluation protocol

## 1. Source

`mvaldenegro/marine-debris-fls-datasets` at commit `4a77998`, folder
`md_fls_dataset/data/watertank-segmentation/` (Valdenegro-Toro et al., 2025):
1,868 forward-looking sonar frames recorded with an ARIS Explorer 3000 in the
Ocean Systems Lab water tank, projected to Cartesian coordinates at 320x480
pixels, single channel, with one Pascal-VOC-style XML per frame giving boxes
as (x, y, w, h). Only `Images/` and `BoxAnnotations/` are used (sparse clone,
246 MB); the segmentation masks are not fetched. All statistics below are
reproduced by `uv run python -m sonar_edge.stats` and `uv run python -m sonar_edge.split`.

## 2. Frames and boxes

Every frame has at least one box (at most 5, mean 2.08); 3,882 boxes in 11
classes. The eleventh class, **Wall** (1,056 boxes in 915 frames), labels the
tank walls. It is retained as a detector class, since a deployed system must
not report walls as debris, but excluded from the scene matching of Section 3
because walls are static and would link unrelated scenes.

| Class | Boxes | Frames | Median w x h (px) | w range | h range |
|---|---|---|---|---|---|
| Bottle | 468 | 416 | 65 x 47 | 16-159 | 36-213 |
| Can | 292 | 292 | 40 x 40 | 30-53 | 20-217 |
| Chain | 320 | 313 | 121 x 149 | 17-239 | 20-253 |
| Drink-carton | 347 | 347 | 27 x 30 | 16-49 | 21-76 |
| Hook | 171 | 170 | 47 x 63 | 36-99 | 41-81 |
| Propeller | 197 | 197 | 69 x 77 | 15-119 | 17-149 |
| Shampoo-bottle | 111 | 109 | 32 x 99 | 20-74 | 22-223 |
| Standing-bottle | 65 | 65 | 25 x 108 | 21-68 | 53-132 |
| Tire | 615 | 615 | 60 x 55 | 18-97 | 28-100 |
| Valve | 240 | 237 | 59 x 45 | 37-299 | 13-129 |
| Wall | 1056 | 915 | 79 x 68 | 10-303 | 12-188 |

## 3. Frame order and scene reconstruction

### 3.1 Frame indices carry no temporal information

A split by recording order would be the natural defence against the
near-duplicate leak documented in the predecessor study. Two tests show that the
frame index is not recording order:

| Pair | Median thumbnail correlation | Box layouts fully matching |
|---|---|---|
| (i, i+1) | 0.757 | 2% |
| (i, i+10) | 0.746 | 3% |
| random | 0.747 | 2% |

Thumbnail correlation uses 32x48 mean-centred, L2-normalised thumbnails; a
layout "fully matches" when every box in one frame has a same-class box in the
other at IoU >= 0.5. Consecutive indices are indistinguishable from random
pairs, and class presence is uniform along the index range. The separate
`watertank-v1.0` release (329 MB) describes no sequence metadata either and was
not fetched.

### 3.2 Scenes

Since the recordings cannot be recovered from metadata, they are reconstructed
from content. Two frames are linked when, ignoring Wall boxes, every box of each
frame has a same-class box in the other at IoU >= 0.5; the connected components
of this graph are called *scenes*. Single linkage follows the slow drift of the
sonar through a recording while the full-match requirement prevents chaining
through a single shared object. The procedure yields **191 scenes** (largest
201, 158, 158, 119 and 90 frames; 118 single-frame scenes, mostly frames whose
only boxes are walls). Visual inspection of random members of four scenes of
different sizes confirmed that each is one physical arrangement of objects seen
through small viewpoint and gain changes.

Including Wall boxes in the matching, or lowering the required fraction of
matched boxes to 0.5, merges 1,686 frames into one component: the static walls
act as a bridge between otherwise unrelated scenes.

### 3.3 Class concentration

Several classes occur almost entirely in one scene: standing-bottle (94% of its
boxes), valve (66%), drink-carton (54%), bottle (43%). No single held-out set
can therefore both train and test on these classes.

## 4. Evaluation protocol

Scenes are assigned to K = 5 folds by a greedy procedure: scenes are taken
largest first, and each goes to the fold that minimises the sum over classes of
the squared share of that class's boxes plus the squared share of frames after
the assignment. The fold file `splits/scene_folds_v1.json` (frame -> scene,
fold) is committed, so the split is fixed.

| | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| frames | 439 | 388 | 383 | 230 | 428 |
| Bottle | 43% | 12% | 6% | 3% | 36% |
| Can | 22% | 18% | 24% | 3% | 33% |
| Chain | 14% | 15% | 22% | 37% | 11% |
| Drink-carton | 24% | 54% | 6% | 2% | 14% |
| Hook | 23% | 30% | 13% | 6% | 28% |
| Propeller | 32% | 23% | 22% | 5% | 18% |
| Shampoo-bottle | 7% | 10% | 35% | 5% | 42% |
| Standing-bottle | 3% | 0% | 3% | 94% | 0% |
| Tire | 34% | 39% | 9% | 3% | 15% |
| Valve | 2% | 17% | 66% | 5% | 10% |
| Wall | 17% | 25% | 16% | 18% | 24% |

- **Development protocol:** fold 0 is the test set, fold 1 the validation set
  (model selection), folds 2-4 the training set (1,041 / 388 / 439 frames).
- **Final protocol:** 5-fold cross-validation; every scene is tested exactly
  once and AP is pooled over folds per class.
- Standing-bottle cannot be evaluated meaningfully under any protocol: when
  fold 3 is held out, the training set contains 4 standing bottles. Its AP is
  reported and flagged.

## 5. Discussion

The reconstruction replaces the missing recording metadata with a conservative
proxy: two frames end up in different folds unless they demonstrably show the
same arrangement. It errs on the side of linking too little (a scene whose
objects were all re-annotated with shifted boxes would split in two), which can
only make the evaluation stricter, not looser. The concentration of some classes
in single scenes is a property of the dataset that random image-level splits
hide entirely; it is the reason the cropped-patch benchmark used by the MCNN
paper can report near-perfect accuracy on classes the network has effectively
seen from a single recording.

## References

- Valdenegro-Toro, M., Padmanabhan, D. C., Singh, D., Wehbe, B. and Petillot, Y.
  (2025). The Marine Debris Forward-Looking Sonar Datasets. OCEANS 2025 Brest.
  arXiv:2503.22880.
