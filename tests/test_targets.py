import numpy as np
import torch

from sonar_edge.metrics import evaluate
from sonar_edge.targets import decode, make_targets


def test_targets_roundtrip_perfect_ap():
    """Build the ideal network output from targets, decode it, and expect AP = 1."""
    stride, gh, gw, C = 8, 60, 40, 11
    boxes = [(0, 10.0, 20.0, 70.0, 60.0), (3, 100.0, 300.0, 130.0, 333.0), (3, 200.0, 50.0, 310.0, 250.0)]
    heat, ind, size, off, mask = make_targets(boxes, C, gh, gw, stride)
    assert mask.sum() == 3 and heat.max() == 1.0
    # logits: large where heat==1, very negative elsewhere
    logits = torch.full((1, C, gh, gw), -10.0)
    logits[0][torch.from_numpy(heat) == 1.0] = 10.0
    size_map = torch.zeros(1, 2, gh, gw)
    off_map = torch.zeros(1, 2, gh, gw)
    for k in range(3):
        iy, ix = divmod(int(ind[k]), gw)
        size_map[0, :, iy, ix] = torch.from_numpy(size[k])
        off_map[0, :, iy, ix] = torch.from_numpy(off[k])
    dets = decode(logits, size_map, off_map, stride, k=10, thr=0.0)[0]
    assert len(dets) == 3
    for c, s, x0, y0, x1, y1 in dets:
        gt = next(b for b in boxes if b[0] == c and abs(b[1] - x0) < 1e-3)
        np.testing.assert_allclose((x0, y0, x1, y1), gt[1:], atol=1e-3)
    aps, m = evaluate({0: dets}, {0: boxes}, C)
    assert m == 1.0 and np.isnan(aps[1])


def test_decode_suppresses_non_peaks():
    logits = torch.full((1, 1, 8, 8), -5.0)
    logits[0, 0, 4, 4] = 3.0
    logits[0, 0, 4, 5] = 2.0  # neighbour, must be suppressed by 3x3 max
    dets = decode(logits, torch.ones(1, 2, 8, 8), torch.zeros(1, 2, 8, 8), 4, k=5, thr=0.0)[0]
    assert len(dets) == 1 and dets[0][1] == 3.0
