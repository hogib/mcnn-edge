"""5-fold cross-validation: pool per-fold test predictions into one AP per class.

Protocol (docs/M0-data.md): fold k is the test set, fold (k+1) mod 5 the
validation set, the rest training. Every scene is tested exactly once, so the
test predictions of the five runs cover the whole dataset without overlap and
can be scored as a single detection problem ("pooled AP", the VOC convention).
This is the only estimate that is interpretable for classes concentrated in a
few scenes (standing-bottle, valve), whose per-fold AP is undefined or rests
on a handful of boxes.

  ./run_cv.sh small_cv small 2 60 basic      # trains + quantizes runs/small_cv_f{0..4}
  uv run sonar-edge cv runs/small_cv         # pools, writes runs/small_cv_cv.json
"""

import json
from pathlib import Path

import numpy as np

from .data import class_names, load_frames
from .metrics import evaluate
from .split import SPLIT_FILE

ROOT = Path(__file__).resolve().parents[2]


def fold_gt(frames, classes, fold, scale, split_file=SPLIT_FILE):
    """image index -> [(cls, x0, y0, x1, y1)] at working scale, for every frame of `fold`."""
    fold_of = {int(i): f for i, f in json.loads(split_file.read_text())["fold"].items()}
    cidx = {c: i for i, c in enumerate(classes)}
    return {
        f.index: [(cidx[b.cls], b.x / scale, b.y / scale, (b.x + b.w) / scale, (b.y + b.h) / scale) for b in f.boxes]
        for f in frames if fold_of[f.index] == fold
    }


def load_pred(path):
    return {int(k): [tuple(d) for d in v] for k, v in json.loads(Path(path).read_text()).items()}


def pool(preds, gts, n_classes):
    """preds/gts: lists of per-fold dicts with disjoint image indices. Returns (per-class AP, mAP)."""
    p, g = {}, {}
    for pp, gg in zip(preds, gts):
        assert not (p.keys() & gg.keys()), "folds overlap"
        p.update(pp)
        g.update(gg)
    assert p.keys() <= g.keys(), "prediction for a frame outside the pooled test sets"
    return evaluate(p, g, n_classes)


def cmd_cv(args):
    import torch

    prefix = ROOT / args.prefix
    frames = load_frames()
    classes = class_names(frames)
    runs = [prefix.parent / f"{prefix.name}_f{k}" for k in range(args.k)]
    missing = [r for r in runs if not (r / "pred_test.json").exists()]
    if missing:
        raise SystemExit("missing fold runs: " + ", ".join(str(m.relative_to(ROOT)) for m in missing))
    scales = {torch.load(r / "best.pt", map_location="cpu", weights_only=False)["scale"] for r in runs}
    assert len(scales) == 1, f"folds trained at different scales: {scales}"
    scale = scales.pop()
    gts = [fold_gt(frames, classes, k, scale) for k in range(args.k)]
    n_boxes = {c: sum(g[0] == i for gg in gts for gs in gg.values() for g in gs) for i, c in enumerate(classes)}

    out = {"prefix": args.prefix, "k": args.k, "scale": scale, "n_test_frames": sum(len(g) for g in gts)}
    variants = {"float": "pred_test.json", "int8": "pred_test_int8.json"}
    for name, fn in variants.items():
        if not all((r / fn).exists() for r in runs):
            continue
        preds = [load_pred(r / fn) for r in runs]
        per_fold = [evaluate(p, g, len(classes))[1] for p, g in zip(preds, gts)]
        aps, m = pool(preds, gts, len(classes))
        out[name] = {
            "map_pooled": m,
            "ap_pooled": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)},
            "map_per_fold": per_fold,
            "map_fold_mean": float(np.mean(per_fold)),
            "map_fold_std": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else None,
        }
    out["n_test_boxes"] = n_boxes
    out["best_epoch_per_fold"] = [json.loads((r / "results.json").read_text())["epoch"] for r in runs]

    cols = [n for n in variants if n in out]
    print(f"{args.k}-fold CV, {args.prefix}, scale {scale}, {out['n_test_frames']} test frames")
    print(f"{'':18s}{'boxes':>7s}" + "".join(f"{c:>9s}" for c in cols))
    for c in classes:
        print(f"{c:18s}{n_boxes[c]:7d}" + "".join(f"{100 * out[v]['ap_pooled'][c]:9.1f}" for v in cols))
    print(f"{'mAP pooled':18s}{'':7s}" + "".join(f"{100 * out[v]['map_pooled']:9.1f}" for v in cols))
    print(f"{'mAP fold mean':18s}{'':7s}" + "".join(f"{100 * out[v]['map_fold_mean']:9.1f}" for v in cols))
    if args.k > 1:
        print(f"{'mAP fold std':18s}{'':7s}" + "".join(f"{100 * out[v]['map_fold_std']:9.1f}" for v in cols))
    for v in cols:
        print(f"  {v} per fold: " + " ".join(f"{100 * x:.1f}" for x in out[v]["map_per_fold"]))
    dest = prefix.parent / f"{prefix.name}_cv.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")
