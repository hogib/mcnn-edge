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
    results = [json.loads((r / "results.json").read_text()) for r in runs]
    out["best_epoch_per_fold"] = [r["epoch"] for r in results]
    if all("seg" in r["test"] for r in results):  # pooled mIoU: sum the per-fold confusion matrices
        from .train import seg_iou

        conf = np.sum([np.array(r["test"]["seg"]["confusion"]) for r in results], axis=0)
        iou = seg_iou(conf)
        names = ["Background"] + classes
        out["seg"] = {"miou_pooled": float(np.nanmean(iou)), "miou_fg_pooled": float(np.nanmean(iou[1:])),
                      "iou_pooled": {c: (None if np.isnan(v) else float(v)) for c, v in zip(names, iou)},
                      "miou_per_fold": [r["test"]["seg"]["miou"] for r in results]}

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
    if "seg" in out:
        print(f"{'segmentation IoU':18s}{'':7s}{'pooled':>9s}")
        for c, v in out["seg"]["iou_pooled"].items():
            print(f"{c:18s}{'':7s}{'n/a' if v is None else f'{100 * v:9.1f}':>9s}")
        print(f"{'mIoU pooled':18s}{'':7s}{100 * out['seg']['miou_pooled']:9.1f}   (foreground {100 * out['seg']['miou_fg_pooled']:.1f})")
        print("  mIoU per fold: " + " ".join(f"{100 * x:.1f}" for x in out["seg"]["miou_per_fold"]))
    dest = prefix.parent / f"{prefix.name}_cv.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")


def seed_summary(cv_files):
    """Aggregate several `_cv.json` files (same configuration, different seeds): per-seed pooled
    mAP, and mean/sd over seeds of the pooled mAP, of each fold's mAP and of each class's pooled AP."""
    runs = [json.loads(Path(f).read_text()) for f in cv_files]
    assert len({r["k"] for r in runs}) == 1 and len({r["scale"] for r in runs}) == 1, "mixed k or scale"
    classes = list(runs[0]["n_test_boxes"])
    out = {"prefixes": [r["prefix"] for r in runs], "n_seeds": len(runs), "k": runs[0]["k"], "scale": runs[0]["scale"]}
    sd = (lambda x: float(np.std(x, ddof=1))) if len(runs) > 1 else (lambda x: None)
    for v in ("float", "int8"):
        if not all(v in r for r in runs):
            continue
        pooled = [r[v]["map_pooled"] for r in runs]
        folds = np.array([r[v]["map_per_fold"] for r in runs])  # seeds x k
        ap = {c: [r[v]["ap_pooled"][c] for r in runs] for c in classes}
        out[v] = {
            "map_pooled_per_seed": pooled,
            "map_pooled_mean": float(np.mean(pooled)), "map_pooled_sd": sd(pooled),
            "map_per_fold_per_seed": folds.tolist(),
            "map_per_fold_mean": folds.mean(0).tolist(),
            "map_per_fold_sd": folds.std(0, ddof=1).tolist() if len(runs) > 1 else None,
            "ap_pooled_mean": {c: (None if any(a is None for a in ap[c]) else float(np.mean(ap[c]))) for c in classes},
            "ap_pooled_sd": {c: (None if any(a is None for a in ap[c]) or len(runs) < 2 else sd(ap[c])) for c in classes},
        }
    return out


def cmd_seeds(args):
    files = [ROOT / f"{p}_cv.json" for p in args.prefixes]
    out = seed_summary(files)
    cols = [v for v in ("float", "int8") if v in out]
    pm = lambda m, s: f"{100 * m:5.1f} +- {100 * s:3.1f}" if s is not None else f"{100 * m:5.1f}"
    print(f"{out['n_seeds']} seeds of {', '.join(out['prefixes'])}")
    for v in cols:
        o = out[v]
        print(f"[{v}] pooled mAP per seed: " + " ".join(f"{100 * x:.1f}" for x in o["map_pooled_per_seed"])
              + f"   mean {pm(o['map_pooled_mean'], o['map_pooled_sd'])}")
        for k in range(out["k"]):
            col = [s[k] for s in o["map_per_fold_per_seed"]]
            print(f"   fold {k}: " + " ".join(f"{100 * x:.1f}" for x in col)
                  + f"   mean {pm(o['map_per_fold_mean'][k], o['map_per_fold_sd'][k] if o['map_per_fold_sd'] else None)}")
    print(f"{'class (pooled AP)':20s}" + "".join(f"{v:>16s}" for v in cols))
    for c in out[cols[0]]["ap_pooled_mean"]:
        print(f"{c:20s}" + "".join(f"{pm(out[v]['ap_pooled_mean'][c], out[v]['ap_pooled_sd'][c]):>16s}" for v in cols))
    dest = ROOT / f"{args.prefixes[0]}_seeds.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")
