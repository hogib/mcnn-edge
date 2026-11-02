"""M4: run the C runtime over a whole split and score it with the same metric
as the Python models. The C mAP must equal the Python int8 mAP.

  uv run python -m sonar_edge.c_eval runs/<name> [--split test] [--bin runtime/sonar-detect] [--repeat N]

Frames are exported once as binary PGM at the working resolution to
runs/<name>/pgm/<split>/; the CLI is run once per batch of files.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .dataset import dev_datasets
from .metrics import evaluate
from .model import Detector

ROOT = Path(__file__).resolve().parents[2]


def export_pgm(ds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(ds.frames):
        p = out_dir / f"{f.index}.pgm"
        if not p.exists():
            img = np.round(ds._prep(ds.images[i]) * 255).astype(np.uint8)
            with open(p, "wb") as fh:
                fh.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
                fh.write(img.tobytes())
        paths.append(p)
    return paths


def run_cli(binary: Path, paths, thr=-4.0, repeat=1, batch=200):
    pred, timing = {}, []
    for a in range(0, len(paths), batch):
        chunk = paths[a : a + batch]
        t0 = time.time()
        r = subprocess.run([str(binary), "-t", str(thr), "-n", str(repeat)] + [str(p) for p in chunk],
                           capture_output=True, text=True, check=True)
        timing.append((time.time() - t0, len(chunk) * repeat))
        for line in r.stdout.splitlines():
            f, c, s, x0, y0, x1, y1 = line.split()
            idx = int(Path(f).stem)
            pred.setdefault(idx, []).append((int(c), float(s), float(x0), float(y0), float(x1), float(y1)))
        if r.stderr:
            print(r.stderr.strip())
    for p in paths:
        pred.setdefault(int(p.stem), [])
    return pred, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bin", default="runtime/sonar-detect")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--test-fold", type=int, default=0)
    ap.add_argument("--val-fold", type=int, default=1)
    args = ap.parse_args()
    run = ROOT / args.run
    import torch
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = Detector(config=ck["config"])
    ds, classes = dev_datasets(model.stride, ck["scale"], args.test_fold, args.val_fold)
    d = ds[args.split]
    paths = export_pgm(d, run / "pgm" / args.split)
    pred, timing = run_cli(ROOT / args.bin, paths, repeat=args.repeat)
    aps, m = evaluate(pred, d.gt_boxes(), len(classes))
    q = json.loads((run / "quant_results.json").read_text())[args.split]
    print(f"{args.split}: C runtime mAP@0.5 {100 * m:.2f}   Python int8 {100 * q['map_int8']:.2f}   "
          f"float {100 * q['map_float']:.2f}")
    worst = 0.0
    for c, a in zip(classes, aps):
        pa = q["ap"][c]
        if pa is None:
            continue
        worst = max(worst, abs(a - pa))
        print(f"   {c:16s} C {100 * a:5.1f}  py {100 * pa:5.1f}")
    print(f"max per-class |C - Python| = {100 * worst:.2f} points")
    n = sum(t[1] for t in timing)
    print(f"wall time incl. process start and PGM I/O: {1e3 * sum(t[0] for t in timing) / n:.2f} ms/frame over {n} frames")
    (run / f"c_eval_{args.split}.json").write_text(json.dumps(
        {"map_c": m, "map_int8_py": q["map_int8"], "ap_c": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)}}, indent=1))


if __name__ == "__main__":
    main()
