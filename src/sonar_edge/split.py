"""Scene-level grouping and split.

Frame indices are shuffled (M0: adjacent frames match layouts 2% of the time,
same as random pairs), so sequences must be reconstructed. Two frames are linked
when their box layouts match (same classes at the same places); connected
components of that graph are "scenes". A scene is held out whole, so the test
set never contains an arrangement the model saw in training.

Some classes are concentrated in one scene (standing-bottle: 94% of its boxes),
so a single split cannot both train and test on them. Scenes are therefore
assigned to K=5 folds; the final evaluation is K-fold cross-validation (every
scene tested once). Development uses fold 0 as test and fold 1 as validation.
"""

import json
from pathlib import Path

SPLIT_FILE = Path(__file__).resolve().parents[2] / "splits/scene_folds_v1.json"

import numpy as np

from .stats import layout_match


def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def scene_groups(frames, thr=1.0, iou_thr=0.5, ignore=("Wall",)):
    """Group id per frame. Link i-j if layout_match(i, j) >= thr in BOTH directions.

    Boxes of classes in `ignore` are left out of the matching: walls are static
    tank features that match across unrelated scenes and chain everything together.
    """
    from .data import Frame

    N = len(frames)
    fr = [Frame(f.index, f.image_path, [b for b in f.boxes if b.cls not in ignore]) for f in frames]
    parent = list(range(N))
    for i in range(N):
        if not fr[i].boxes:
            continue
        for j in range(i + 1, N):
            if not fr[j].boxes:
                continue
            if layout_match(fr[i], fr[j], iou_thr) >= thr and layout_match(fr[j], fr[i], iou_thr) >= thr:
                ri, rj = _find(parent, i), _find(parent, j)
                if ri != rj:
                    parent[ri] = rj
    roots = [_find(parent, i) for i in range(N)]
    ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([ids[r] for r in roots])


def describe_groups(groups, label=""):
    sizes = np.bincount(groups)
    sizes = np.sort(sizes)[::-1]
    print(f"{label}: {len(sizes)} groups; largest {sizes[:8].tolist()}; "
          f"singletons {(sizes == 1).sum()}; frames in groups >= 5: {sizes[sizes >= 5].sum()}")


def make_folds(frames, groups, classes, k=5):
    """Fold id per frame. Groups are taken largest first and each goes to the fold
    that minimises the squared per-class box shares plus squared frame share
    after adding it (so a fold that already holds much of a class, or many
    frames, is penalised). Deterministic."""
    cidx = {c: j for j, c in enumerate(classes)}
    ng = groups.max() + 1
    gc = np.zeros((ng, len(classes)), float)
    gsize = np.bincount(groups, minlength=ng).astype(float)
    for f, g in zip(frames, groups):
        for b in f.boxes:
            gc[g, cidx[b.cls]] += 1
    total = np.maximum(gc.sum(0), 1)
    fold_c = np.zeros((k, len(classes)), float)
    fold_n = np.zeros(k)
    gfold = np.zeros(ng, int)
    for g in np.argsort(-gsize, kind="stable"):
        best, best_cost = None, None
        for f in range(k):
            share = (fold_c[f] + gc[g]) / total
            cost = float((share**2).sum()) + ((fold_n[f] + gsize[g]) / len(frames)) ** 2 * len(classes)
            if best_cost is None or cost < best_cost:
                best, best_cost = f, cost
        gfold[g] = best
        fold_c[best] += gc[g]
        fold_n[best] += gsize[g]
    return gfold[groups]


def fold_report(frames, folds, classes, k=5):
    print(f"{'':18s}" + "".join(f"{'fold ' + str(f):>9s}" for f in range(k)))
    print(f"{'frames':18s}" + "".join(f"{(folds == f).sum():9d}" for f in range(k)))
    for c in classes:
        cnt = np.zeros(k, int)
        for f, fo in zip(frames, folds):
            cnt[fo] += sum(b.cls == c for b in f.boxes)
        print(f"{c:18s}" + "".join(f"{v:5d} {100 * v / cnt.sum():2.0f}%" for v in cnt))


def load_split(dev_test=0, dev_val=1):
    """frame index -> 'train' | 'val' | 'test' for the development split."""
    d = json.loads(SPLIT_FILE.read_text())
    out = {}
    for idx, fo in d["fold"].items():
        out[int(idx)] = "test" if fo == dev_test else "val" if fo == dev_val else "train"
    return out


if __name__ == "__main__":
    from .data import class_names, load_frames

    frames = load_frames()
    classes = class_names(frames)
    for ignore in ((), ("Wall",)):
        for thr in (1.0, 0.75, 0.5):
            describe_groups(scene_groups(frames, thr, ignore=ignore), f"ignore={ignore} thr {thr}")
    groups = scene_groups(frames, 1.0)
    print("\nclass: groups containing it / largest group's share of its boxes")
    for c in classes:
        per = np.zeros(groups.max() + 1)
        for f, g in zip(frames, groups):
            per[g] += sum(b.cls == c for b in f.boxes)
        print(f"  {c:18s} {(per > 0).sum():4d} groups  max share {per.max() / per.sum():.2f}")
    folds = make_folds(frames, groups, classes)
    print()
    fold_report(frames, folds, classes)
    SPLIT_FILE.parent.mkdir(exist_ok=True)
    SPLIT_FILE.write_text(json.dumps({
        "method": "scene groups: box-layout match thr 1.0 (IoU 0.5), Wall ignored; balanced greedy K=5 folds",
        "n_groups": int(groups.max() + 1), "k": 5,
        "group": {int(f.index): int(g) for f, g in zip(frames, groups)},
        "fold": {int(f.index): int(fo) for f, fo in zip(frames, folds)},
    }, indent=0))
    print(f"wrote {SPLIT_FILE}")


