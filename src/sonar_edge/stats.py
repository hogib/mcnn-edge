"""M0 dataset statistics: classes, boxes, frame order, class presence along the recording.

  uv run python -m sonar_edge.stats
"""

import numpy as np

from .data import class_names, load_frames


def adjacent_similarity(frames, step=1, n=400, seed=0):
    """Correlation of 32x32 thumbnails between frame i and i+step (or random pairs for step=None)."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    N = len(frames)

    def thumb(f):
        v = np.asarray(Image.open(f.image_path).convert("L").resize((32, 48)), np.float32).ravel()
        v -= v.mean()
        return v / (np.linalg.norm(v) + 1e-8)

    sims = []
    for _ in range(n):
        i = int(rng.integers(0, N - (step or 1)))
        j = i + step if step else int(rng.integers(0, N))
        if i == j:
            continue
        sims.append(float(thumb(frames[i]) @ thumb(frames[j])))
    return np.array(sims)


def main():
    frames = load_frames()
    classes = class_names(frames)
    print(f"{len(frames)} frames, {sum(len(f.boxes) for f in frames)} boxes, {len(classes)} classes")
    print(f"frames with no boxes: {sum(not f.boxes for f in frames)}")
    print(f"boxes per frame: max {max(len(f.boxes) for f in frames)}, "
          f"mean {np.mean([len(f.boxes) for f in frames]):.2f}")

    print("\nPer class: boxes, frames, median w x h, w range, h range")
    for c in classes:
        bs = [b for f in frames for b in f.boxes if b.cls == c]
        nf = sum(any(b.cls == c for b in f.boxes) for f in frames)
        w = np.array([b.w for b in bs])
        h = np.array([b.h for b in bs])
        print(f"  {c:18s} {len(bs):5d} {nf:5d}  {np.median(w):4.0f} x {np.median(h):4.0f}  "
              f"w {w.min()}-{w.max()}  h {h.min()}-{h.max()}")

    print("\nFrame order check (thumbnail correlation):")
    for label, step in [("adjacent (i, i+1)", 1), ("i, i+10", 10), ("random pairs", None)]:
        s = adjacent_similarity(frames, step)
        print(f"  {label:18s} median {np.median(s):.3f}  >0.9: {(s > 0.9).mean():.0%}")

    layout_check(frames)

    print("\nClass presence per 10% block of the recording (frames with >=1 box of class):")
    N = len(frames)
    edges = np.linspace(0, N, 11).astype(int)
    print(f"  {'class':18s}" + "".join(f"{k * 10:>6d}%" for k in range(10)))
    for c in classes:
        row = []
        for a, b in zip(edges[:-1], edges[1:]):
            row.append(sum(any(x.cls == c for x in f.boxes) for f in frames[a:b]))
        print(f"  {c:18s}" + "".join(f"{v:7d}" for v in row))


def layout_match(a, b, iou_thr=0.5):
    """Fraction of boxes in frame a with a same-class box in frame b at IoU >= thr."""
    if not a.boxes:
        return 0.0
    hits = 0
    for p in a.boxes:
        for q in b.boxes:
            if p.cls != q.cls:
                continue
            ax0, ay0, ax1, ay1 = p.xyxy
            bx0, by0, bx1, by1 = q.xyxy
            iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih
            union = p.w * p.h + q.w * q.h - inter
            if union and inter / union >= iou_thr:
                hits += 1
                break
    return hits / len(a.boxes)


def layout_check(frames, n=600, seed=0):
    rng = np.random.default_rng(seed)
    N = len(frames)
    print("\nBox-layout match (same class, IoU>=0.5) between frame pairs:")
    for label, step in [("adjacent (i, i+1)", 1), ("i, i+10", 10), ("random pairs", None)]:
        m = []
        for _ in range(n):
            i = int(rng.integers(0, N - (step or 1)))
            j = i + step if step else int(rng.integers(0, N))
            if i != j:
                m.append(layout_match(frames[i], frames[j]))
        m = np.array(m)
        print(f"  {label:18s} mean {m.mean():.3f}  full match: {(m == 1).mean():.0%}  no match: {(m == 0).mean():.0%}")


if __name__ == "__main__":
    main()
