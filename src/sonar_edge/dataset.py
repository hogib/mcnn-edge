"""Torch dataset over whole frames with CenterNet targets.

All frames are preloaded as uint8 (1868 x 480 x 320 = 287 MB). Augmentation
(train only): horizontal flip, gain/offset jitter, small integer shift.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FRAME_H, FRAME_W, class_names, load_frames
from .split import load_split
from .targets import make_targets


class FrameDataset(Dataset):
    def __init__(self, frames, classes, stride, scale=1, train=False, seed=0, max_objs=16):
        self.frames = frames
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.n_classes = len(classes)
        self.stride, self.scale, self.train = stride, scale, train
        self.max_objs = max_objs
        self.rng = np.random.default_rng(seed)
        self.images = np.stack([f.load() for f in frames])  # N,H,W uint8
        self.h, self.w = FRAME_H // scale, FRAME_W // scale
        self.gh, self.gw = self.h // stride, self.w // stride

    def __len__(self):
        return len(self.frames)

    def _prep(self, img):
        """uint8 HxW -> float [0,1] at working scale (area downsample)."""
        if self.scale > 1:
            s = self.scale
            img = img[: self.h * s, : self.w * s].reshape(self.h, s, self.w, s).mean((1, 3))
        return img.astype(np.float32) / 255.0

    def __getitem__(self, i):
        f = self.frames[i]
        img = self._prep(self.images[i])
        boxes = [(self.cidx[b.cls], b.x / self.scale, b.y / self.scale,
                  (b.x + b.w) / self.scale, (b.y + b.h) / self.scale) for b in f.boxes]
        if self.train:
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
                boxes = [(c, self.w - x1, y0, self.w - x0, y1) for c, x0, y0, x1, y1 in boxes]
            dx, dy = (int(v) for v in self.rng.integers(-8 // self.scale, 8 // self.scale + 1, size=2))
            if dx or dy:
                img = np.roll(img, (dy, dx), axis=(0, 1))
                boxes = [(c, np.clip(x0 + dx, 0, self.w), np.clip(y0 + dy, 0, self.h),
                          np.clip(x1 + dx, 0, self.w), np.clip(y1 + dy, 0, self.h)) for c, x0, y0, x1, y1 in boxes]
                boxes = [b for b in boxes if b[3] - b[1] >= 2 and b[4] - b[2] >= 2]
            gain = self.rng.uniform(0.7, 1.3)
            bias = self.rng.uniform(-0.1, 0.1)
            img = np.clip(img * gain + bias, 0, 1)
        img = np.ascontiguousarray(img, np.float32)
        heat, ind, size, off, mask = make_targets(boxes, self.n_classes, self.gh, self.gw, self.stride, self.max_objs)
        return {
            "image": torch.from_numpy(img)[None],
            "heat": torch.from_numpy(heat),
            "ind": torch.from_numpy(ind),
            "size": torch.from_numpy(size),
            "off": torch.from_numpy(off),
            "mask": torch.from_numpy(mask),
            "index": f.index,
        }

    def gt_boxes(self):
        """image index -> list of (cls, x0, y0, x1, y1) at working scale, for evaluation."""
        return {
            f.index: [(self.cidx[b.cls], b.x / self.scale, b.y / self.scale,
                       (b.x + b.w) / self.scale, (b.y + b.h) / self.scale) for b in f.boxes]
            for f in self.frames
        }


def dev_datasets(stride, scale=1, test_fold=0, val_fold=1, seed=0):
    frames = load_frames()
    classes = class_names(frames)
    split = load_split(test_fold, val_fold)
    parts = {k: [f for f in frames if split[f.index] == k] for k in ("train", "val", "test")}
    ds = {k: FrameDataset(v, classes, stride, scale, train=(k == "train"), seed=seed) for k, v in parts.items()}
    return ds, classes
