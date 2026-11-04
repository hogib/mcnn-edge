"""Watertank full-frame dataset: 320x480 single-channel sonar frames with boxes.

Frames are numbered 0..N-1 in recording order (verified in M0 by adjacent-frame
similarity). Boxes come from Pascal-VOC-style XML with COCO x y w h fields.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data/raw/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation"
)
FRAME_W, FRAME_H = 320, 480


@dataclass
class Box:
    cls: str
    x: int
    y: int
    w: int
    h: int

    @property
    def xyxy(self):
        return self.x, self.y, self.x + self.w, self.y + self.h


@dataclass
class Frame:
    index: int
    image_path: Path
    boxes: list[Box]
    mask_path: Path | None = None  # semantic label PNG: 0 background, 1..11 = classes in sorted order

    def load(self) -> np.ndarray:
        """HxW uint8."""
        return np.asarray(Image.open(self.image_path).convert("L"))

    def load_mask(self) -> np.ndarray:
        """HxW uint8 label map (0 = background, c+1 = class index c of class_names)."""
        assert self.mask_path is not None, f"no mask for frame {self.index} (fetch Masks/ into the sparse checkout)"
        return np.asarray(Image.open(self.mask_path))


def _frame_index(path: Path) -> int:
    return int(re.search(r"-(\d+)$", path.stem)[1])


def parse_xml(path: Path) -> tuple[str, tuple[int, int], list[Box]]:
    root = ET.parse(path).getroot()
    fn = root.findtext("filename")
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height")))
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append(Box(obj.findtext("name").strip(), *(int(float(b.findtext(k))) for k in "xywh")))
    return fn, wh, boxes


def load_frames(root: Path = DATA_ROOT) -> list[Frame]:
    """All frames in index order, with boxes (possibly empty)."""
    frames = []
    for xml in sorted((root / "BoxAnnotations").glob("*.xml"), key=_frame_index):
        fn, wh, boxes = parse_xml(xml)
        assert wh == (FRAME_W, FRAME_H), (xml, wh)
        img = root / "Images" / fn
        assert img.exists(), img
        mask = root / "Masks" / fn
        frames.append(Frame(_frame_index(xml), img, boxes, mask if mask.exists() else None))
    assert [f.index for f in frames] == list(range(len(frames))), "non-contiguous frame indices"
    return frames


def class_names(frames: list[Frame]) -> list[str]:
    return sorted({b.cls for f in frames for b in f.boxes})
