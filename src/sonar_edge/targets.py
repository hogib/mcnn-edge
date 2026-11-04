"""CenterNet targets and decoding, shared by training, evaluation and the
golden-vector export (so the C decoder is checked against exactly this)."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_radius(h, w, min_overlap=0.7):
    """CornerNet/CenterNet radius so that a box shifted by r still has IoU >= min_overlap."""
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(heat, cx, cy, radius):
    """Max-splat a 2D gaussian of given integer radius at integer (cx, cy)."""
    d = 2 * radius + 1
    sigma = d / 6
    ys, xs = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    g = np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma))
    H, W = heat.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    patch = heat[cy - t : cy + b, cx - l : cx + r]
    np.maximum(patch, g[radius - t : radius + b, radius - l : radius + r], out=patch)


def make_targets(boxes, n_classes, grid_h, grid_w, stride, max_objs=16):
    """boxes: list of (cls_idx, x0, y0, x1, y1) in input pixels.
    Returns heat [C,gh,gw], and per-object arrays (index into grid, size, offset, mask)."""
    heat = np.zeros((n_classes, grid_h, grid_w), np.float32)
    ind = np.zeros(max_objs, np.int64)
    size = np.zeros((max_objs, 2), np.float32)
    off = np.zeros((max_objs, 2), np.float32)
    mask = np.zeros(max_objs, np.float32)
    for k, (c, x0, y0, x1, y1) in enumerate(boxes[:max_objs]):
        w, h = (x1 - x0) / stride, (y1 - y0) / stride
        cx, cy = (x0 + x1) / 2 / stride, (y0 + y1) / 2 / stride
        ix, iy = min(int(cx), grid_w - 1), min(int(cy), grid_h - 1)
        r = max(0, int(gaussian_radius(math.ceil(h), math.ceil(w))))
        draw_gaussian(heat[c], ix, iy, r)
        ind[k] = iy * grid_w + ix
        size[k] = (w, h)
        off[k] = (cx - ix, cy - iy)
        mask[k] = 1
    return heat, ind, size, off, mask


def focal_loss(logits, target):
    """Penalty-reduced pixelwise focal loss (CenterNet), normalised by #objects."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    neg_w = (1 - target).pow(4)
    pos_loss = (torch.log(p) * (1 - p).pow(2) * pos).sum()
    neg_loss = (torch.log(1 - p) * p.pow(2) * neg_w * neg).sum()
    n = pos.sum().clamp(min=1)
    return -(pos_loss + neg_loss) / n


def dice_loss(logits, target, eps=1.0):
    """Soft multi-class Dice: 1 - (2 |p ∩ t| + eps) / (|p| + |t| + eps) per class on the softmax
    probabilities, averaged over the classes present in the batch (background included)."""
    p = torch.softmax(logits, 1)
    t = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).to(p.dtype)
    inter = (p * t).sum((0, 2, 3))
    card = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    present = t.sum((0, 2, 3)) > 0
    d = 1 - (2 * inter + eps) / (card + eps)
    return d[present].mean()


def gather(feat, ind):
    """feat [B,2,gh,gw], ind [B,K] -> [B,K,2]."""
    B, C = feat.shape[:2]
    f = feat.view(B, C, -1).permute(0, 2, 1)
    return f.gather(1, ind.unsqueeze(2).expand(-1, -1, C))


def reg_l1(pred, ind, target, mask):
    p = gather(pred, ind)
    m = mask.unsqueeze(2)
    return (torch.abs(p - target) * m).sum() / (m.sum() * 2).clamp(min=1)


def decode(heat_logits, size, off, stride, k=32, thr=0.0):
    """Peaks of the heatmap (3x3 max suppression) -> boxes.
    Works on logits: sigmoid is monotone, so the C decoder can threshold logits too.
    Returns per image a list of (cls, score_logit, x0, y0, x1, y1), descending score."""
    B, C, gh, gw = heat_logits.shape
    peak = F.max_pool2d(heat_logits, 3, 1, 1)
    keep = (peak == heat_logits).float()
    h = heat_logits * keep + (keep - 1) * 1e9
    scores, idx = h.view(B, -1).topk(k)
    out = []
    for b in range(B):
        dets = []
        for s, i in zip(scores[b].tolist(), idx[b].tolist()):
            if s < thr:
                break
            c, cell = divmod(i, gh * gw)
            iy, ix = divmod(cell, gw)
            w, hh = size[b, :, iy, ix].tolist()
            dx, dy = off[b, :, iy, ix].tolist()
            cx, cy = (ix + dx) * stride, (iy + dy) * stride
            dets.append((c, s, cx - w * stride / 2, cy - hh * stride / 2, cx + w * stride / 2, cy + hh * stride / 2))
        out.append(dets)
    return out
