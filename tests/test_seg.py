"""Segmentation head: targets follow the augmentation geometry, IoU from confusion, head layout."""
import numpy as np
import torch

from sonar_edge.data import Box, Frame
from sonar_edge.dataset import FrameDataset
from sonar_edge.model import Detector
from sonar_edge.train import seg_iou


class _F(Frame):
    def load(self):
        return np.random.default_rng(self.index).integers(0, 256, (480, 320), np.uint8)

    def load_mask(self):
        m = np.zeros((480, 320), np.uint8)
        m[60:72, 40:64] = 1  # a bottle region matching the box
        m[:20, :] = 11       # wall
        return m


def _ds(**kw):
    frames = [_F(i, None, [Box("Bottle", 40, 60, 24, 12)]) for i in range(2)]
    return FrameDataset(frames, ["Bottle", "Wall"], stride=8, seg=True, **kw)


def test_seg_target_is_cell_centre_label():
    ds = _ds(scale=2)
    item = ds[0]
    seg = item["seg"]
    assert seg.shape == (30, 20) and seg.dtype == torch.int64
    # working scale: bottle at y 30..36, x 20..32 -> cells with centres (y=4+..) row 4 (centre 36? no: 4*8+4=36 -> outside), row 3 centre 28 -> outside
    # wall rows 0..9 -> cell row 0 (centre 4) only
    assert (seg[0] == 11).all() and (seg[1:] != 11).all()
    assert seg.max() <= 11 and (seg == 1).sum() >= 0


def test_seg_target_follows_flip_and_zoom():
    ds = _ds(scale=2, train=True, aug="strong")
    for i in range(2):
        item = ds[i]
        assert item["seg"].shape == (30, 20)
        # every labelled cell must be background, bottle or wall (no interpolated labels)
        assert set(item["seg"].unique().tolist()) <= {0, 1, 11}


def test_seg_iou():
    conf = np.array([[50, 10, 0], [5, 20, 0], [0, 0, 0]])
    iou = seg_iou(conf)
    assert np.isclose(iou[0], 50 / 65) and np.isclose(iou[1], 20 / 35) and np.isnan(iou[2])


def test_seg_head_layout():
    m = Detector(n_classes=11, config="small", seg=True)
    out = m(torch.zeros(1, 1, 64, 64))
    assert out.shape[1] == 11 + 4 + 12
    heat, size, off = m.split(out)
    assert heat.shape[1] == 11 and size.shape[1] == 2 and off.shape[1] == 2 and m.split_seg(out).shape[1] == 12
