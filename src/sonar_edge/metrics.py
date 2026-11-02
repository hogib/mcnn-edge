"""VOC-style AP@0.5 per class and mAP. Own implementation so the C pipeline's
output can be scored with exactly the same code."""

import numpy as np


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def average_precision(dets, gts, iou_thr=0.5):
    """dets: list of (image_id, score, x0, y0, x1, y1); gts: dict image_id -> list of boxes.
    All-point interpolated AP (VOC2010+)."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan")
    dets = sorted(dets, key=lambda d: -d[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(dets))
    for i, (img, _, *box) in enumerate(dets):
        g = gts.get(img, [])
        best, bj = 0.0, -1
        for j, gb in enumerate(g):
            o = iou(box, gb)
            if o > best:
                best, bj = o, j
        if best >= iou_thr and not used[img][bj]:
            used[img][bj] = True
            tp[i] = 1
    fp = 1 - tp
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(pred, gt, n_classes, iou_thr=0.5):
    """pred: dict image_id -> list of (cls, score, x0, y0, x1, y1)
    gt:   dict image_id -> list of (cls, x0, y0, x1, y1)
    Returns (per-class AP array with nan for absent classes, mAP over present classes)."""
    aps = np.full(n_classes, np.nan)
    for c in range(n_classes):
        dets = [(img, d[1], *d[2:]) for img, ds in pred.items() for d in ds if d[0] == c]
        gts = {img: [g[1:] for g in gs if g[0] == c] for img, gs in gt.items()}
        aps[c] = average_precision(dets, gts, iou_thr)
    return aps, float(np.nanmean(aps))
