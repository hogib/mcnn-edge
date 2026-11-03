"""Fractional input scale (e.g. 4/3 -> 360x240) through FrameDataset and the train CLI parser."""
import numpy as np

from sonar_edge.data import Box, Frame
from sonar_edge.dataset import FrameDataset
from sonar_edge.train import _scale


class _F(Frame):
    def load(self):
        return np.random.default_rng(self.index).integers(0, 256, (480, 320), np.uint8)


def _frames(n=2):
    return [_F(i, None, [Box("Bottle", 40, 60, 24, 12)]) for i in range(n)]


def test_scale_parser():
    assert _scale("1") == 1 and isinstance(_scale("2"), int) and _scale("2.0") == 2
    assert abs(_scale("4/3") - 4 / 3) < 1e-9 and _scale("1.5") == 1.5


def test_fractional_scale_shapes_and_boxes():
    ds = FrameDataset(_frames(), ["Bottle"], stride=8, scale=4 / 3, train=False)
    assert (ds.h, ds.w, ds.gh, ds.gw) == (360, 240, 45, 30)
    item = ds[0]
    assert tuple(item["image"].shape) == (1, 360, 240)
    img = ds._prep(ds.images[0])
    assert img.min() >= 0 and img.max() <= 1
    (c, x0, y0, x1, y1), = ds.gt_boxes()[0]
    assert (c, x0, y0, x1, y1) == (0, 30, 45, 48, 54)


def test_fractional_scale_train_item_runs():
    ds = FrameDataset(_frames(), ["Bottle"], stride=8, scale=4 / 3, train=True, aug="strong")
    for i in range(2):
        assert tuple(ds[i]["image"].shape) == (1, 360, 240)


def test_integer_scale_path_unchanged():
    ds = FrameDataset(_frames(1), ["Bottle"], stride=8, scale=2)
    img = ds.images[0]
    ref = img[:240 * 2, :160 * 2].reshape(240, 2, 160, 2).mean((1, 3)).astype(np.float32) / 255.0
    assert np.array_equal(ds._prep(img), ref)
