# M0: data

Source: `mvaldenegro/marine-debris-fls-datasets` @ `4a77998`, folder
`md_fls_dataset/data/watertank-segmentation/` (`Images/`, `BoxAnnotations/`;
masks not fetched). Sparse clone in `data/raw/`, 246 MB.

Regenerate everything here with `uv run python -m sonar_edge.stats` and
`uv run python -m sonar_edge.split`.

## Frames and boxes

1,868 frames, 320x480, 1 channel (cartesian-projected fan, black outside the fan).
3,882 boxes, 11 classes, every frame has at least one box, at most 5, mean 2.08.
The 11th class is **Wall** (1,056 boxes, 915 frames): tank walls, static across
scenes. It is kept as a detector class (it is what a deployed system must not
confuse with debris) but is excluded from scene grouping (below).

| Class | Boxes | Frames | Median w x h | w range | h range |
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

Smallest objects are drink cartons (~27x30): a stride-4 heatmap (80x120 cells)
resolves them comfortably.

## Frame indices are shuffled

Two tests, both comparing frame i with i+1, with i+10, and with a random frame:

| Pair | Thumbnail correlation (median) | Box layouts fully matching (same classes, IoU >= 0.5) |
|---|---|---|
| i, i+1 | 0.757 | 2% |
| i, i+10 | 0.746 | 3% |
| random | 0.747 | 2% |

Consecutive indices are no more related than random pairs, and class presence
is flat across the index range. The index carries no sequence information, so a
"split by recording order" is impossible from this release. (The separate
`watertank-v1.0` release, 329 MB, lists no sequence metadata in its description
either; not fetched.)

## Scenes

Two frames are linked when every non-Wall box in each has a same-class box in
the other at IoU >= 0.5; connected components are scenes. This reconstructs the
recordings: a scene is one physical arrangement of objects seen through sonar
drift and gain changes (verified visually on four groups of different sizes).

- Linking on all boxes including Wall at a looser threshold (0.5) merges 1,686
  frames into one component through the static walls; with Wall ignored and a
  full match required, there are **191 scenes** (largest 201, 158, 158, 119, 90
  frames; 118 single-frame scenes, mostly frames whose only boxes are walls).
- Several classes live mostly in one scene: standing-bottle 94% of its boxes,
  valve 66%, drink-carton 54%, bottle 43%.

## Folds

Scenes are assigned to 5 folds, largest first, each to the fold that minimises
the sum of squared per-class box shares plus squared frame share. File:
`splits/scene_folds_v1.json` (frame index -> scene id and fold).

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

Protocol:
- **Development:** fold 0 = test, fold 1 = validation, folds 2-4 = train.
- **Final numbers:** 5-fold cross-validation, every scene tested once; AP is
  pooled over folds per class.
- Standing-bottle AP will be close to meaningless in any protocol: when fold 3
  is the test fold the model has seen 4 standing bottles. Report it, flag it.
