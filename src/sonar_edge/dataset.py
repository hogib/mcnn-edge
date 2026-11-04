"""Torch dataset over whole frames with CenterNet targets.

All frames are preloaded as uint8 (1868 x 480 x 320 = 287 MB). Augmentation
(train only): horizontal flip, gain/offset jitter, small integer shift.
"""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data import FRAME_H, FRAME_W, class_names, load_frames
from .split import load_split
from .targets import make_targets


class FrameDataset(Dataset):
    def __init__(self, frames, classes, stride, scale=1, train=False, seed=0, max_objs=16, aug="basic", seg=False,
                 seg_up=1):
        self.frames = frames
        self.aug = aug
        self.seg, self.seg_up = seg, seg_up
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.n_classes = len(classes)
        self.stride, self.scale, self.train = stride, scale, train
        self.max_objs = max_objs
        self.rng = np.random.default_rng(seed)
        self.images = np.stack([f.load() for f in frames])  # N,H,W uint8
        self.masks = np.stack([f.load_mask() for f in frames]) if seg else None  # N,H,W uint8 labels
        self.h, self.w = int(round(FRAME_H / scale)), int(round(FRAME_W / scale))  # 480x320 / scale
        self.gh, self.gw = self.h // stride, self.w // stride

    def __len__(self):
        return len(self.frames)

    def _prep(self, img):
        """uint8 HxW -> float [0,1] at working scale (area downsample)."""
        if isinstance(self.scale, int) and self.scale > 1:
            s = self.scale
            img = img[: self.h * s, : self.w * s].reshape(self.h, s, self.w, s).mean((1, 3))
        elif self.scale != 1:  # non-integer factor (e.g. 4/3 -> 360x240): box-filter resize, result uint8
            img = np.asarray(Image.fromarray(img).resize((self.w, self.h), Image.BOX))
        return img.astype(np.float32) / 255.0

    def seg_mask(self, i):
        """Label map at working scale (nearest), HxW uint8; 0 = background, c+1 = class c."""
        m = self.masks[i]
        if self.scale != 1:
            m = np.asarray(Image.fromarray(m).resize((self.w, self.h), Image.NEAREST))
        return m

    def __getitem__(self, i):
        f = self.frames[i]
        img = self._prep(self.images[i])
        seg = self.seg_mask(i) if self.seg else None
        boxes = [(self.cidx[b.cls], b.x / self.scale, b.y / self.scale,
                  (b.x + b.w) / self.scale, (b.y + b.h) / self.scale) for b in f.boxes]
        if self.train:
            if self.rng.random() < 0.5:
                img = img[:, ::-1]
                boxes = [(c, self.w - x1, y0, self.w - x0, y1) for c, x0, y0, x1, y1 in boxes]
                seg = seg[:, ::-1] if seg is not None else None
            j = max(1, int(8 / self.scale))
            dx, dy = (int(v) for v in self.rng.integers(-j, j + 1, size=2))
            if dx or dy:
                img = np.roll(img, (dy, dx), axis=(0, 1))
                seg = np.roll(seg, (dy, dx), axis=(0, 1)) if seg is not None else None
                boxes = [(c, np.clip(x0 + dx, 0, self.w), np.clip(y0 + dy, 0, self.h),
                          np.clip(x1 + dx, 0, self.w), np.clip(y1 + dy, 0, self.h)) for c, x0, y0, x1, y1 in boxes]
                boxes = [b for b in boxes if b[3] - b[1] >= 2 and b[4] - b[2] >= 2]
            if self.aug == "strong" and self.rng.random() < 0.7:
                img, boxes, seg = self._zoom(img, boxes, seg)
            if self.aug == "strong":
                img = img + self.rng.normal(0, self.rng.uniform(0, 0.04), img.shape).astype(np.float32)
            gain = self.rng.uniform(0.7, 1.3)
            bias = self.rng.uniform(-0.1, 0.1)
            img = np.clip(img * gain + bias, 0, 1)
        img = np.ascontiguousarray(img, np.float32)
        heat, ind, size, off, mask = make_targets(boxes, self.n_classes, self.gh, self.gw, self.stride, self.max_objs)
        item = {}
        if seg is not None:  # per-cell label: the label at the cell centre (on the seg_up-times finer grid)
            st = self.stride // self.seg_up
            ys = np.arange(self.gh * self.seg_up) * st + st // 2
            xs = np.arange(self.gw * self.seg_up) * st + st // 2
            item["seg"] = torch.from_numpy(np.ascontiguousarray(seg[ys][:, xs]).astype(np.int64))
        return item | {
            "image": torch.from_numpy(img)[None],
            "heat": torch.from_numpy(heat),
            "ind": torch.from_numpy(ind),
            "size": torch.from_numpy(size),
            "off": torch.from_numpy(off),
            "mask": torch.from_numpy(mask),
            "index": f.index,
        }

    def _zoom(self, img, boxes, seg=None):
        """Random zoom by s in [0.75, 1.3] about a random point, output stays h x w (zero fill).
        `seg` (uint8 label map, optional) gets the same geometry with nearest interpolation."""
        s = float(self.rng.uniform(0.75, 1.3))
        H, W = img.shape
        zh, zw = max(8, int(round(H * s))), max(8, int(round(W * s)))
        z = torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None],
                                            size=(zh, zw), mode="bilinear", align_corners=False)[0, 0].numpy()
        out = np.zeros((H, W), np.float32)
        # offset of the zoomed image's origin in output coords
        ox = int(self.rng.integers(min(0, W - zw), max(0, W - zw) + 1))
        oy = int(self.rng.integers(min(0, H - zh), max(0, H - zh) + 1))
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        cw, ch = min(zw - sx0, W - dx0), min(zh - sy0, H - dy0)
        out[dy0 : dy0 + ch, dx0 : dx0 + cw] = z[sy0 : sy0 + ch, sx0 : sx0 + cw]
        seg_out = None
        if seg is not None:
            zs = np.asarray(Image.fromarray(seg).resize((zw, zh), Image.NEAREST))
            seg_out = np.zeros((H, W), np.uint8)
            seg_out[dy0 : dy0 + ch, dx0 : dx0 + cw] = zs[sy0 : sy0 + ch, sx0 : sx0 + cw]
        nb = []
        for c, x0, y0, x1, y1 in boxes:
            bx0, by0, bx1, by1 = (np.clip(x0 * s + ox, 0, W), np.clip(y0 * s + oy, 0, H),
                                  np.clip(x1 * s + ox, 0, W), np.clip(y1 * s + oy, 0, H))
            if bx1 - bx0 >= 4 and by1 - by0 >= 4 and (bx1 - bx0) * (by1 - by0) >= 0.3 * (x1 - x0) * (y1 - y0) * s * s:
                nb.append((c, float(bx0), float(by0), float(bx1), float(by1)))
        return out, nb, seg_out

    def gt_boxes(self):
        """image index -> list of (cls, x0, y0, x1, y1) at working scale, for evaluation."""
        return {
            f.index: [(self.cidx[b.cls], b.x / self.scale, b.y / self.scale,
                       (b.x + b.w) / self.scale, (b.y + b.h) / self.scale) for b in f.boxes]
            for f in self.frames
        }


def dev_datasets(stride, scale=1, test_fold=0, val_fold=1, seed=0, aug="basic", seg=False, seg_up=1):
    frames = load_frames()
    classes = class_names(frames)
    split = load_split(test_fold, val_fold)
    parts = {k: [f for f in frames if split[f.index] == k] for k in ("train", "val", "test")}
    ds = {k: FrameDataset(v, classes, stride, scale, train=(k == "train"), seed=seed, aug=aug, seg=seg, seg_up=seg_up)
          for k, v in parts.items()}
    return ds, classes
