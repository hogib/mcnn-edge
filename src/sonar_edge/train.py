"""M1: train and evaluate the float detector.

  uv run sonar-edge train --config base --epochs 60 --out runs/base_f0
  uv run sonar-edge eval runs/base_f0/best.pt [--split test]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import dev_datasets
from .metrics import evaluate
from .model import Detector, macs
from .targets import decode, focal_loss, reg_l1

ROOT = Path(__file__).resolve().parents[2]


def predict(model, ds, device, bs=16, k=32, thr=-4.0):
    """image index -> list of (cls, score_logit, x0, y0, x1, y1)."""
    model.eval()
    dl = DataLoader(ds, bs, shuffle=False, num_workers=0)
    out = {}
    with torch.no_grad():
        for batch in dl:
            o = model(batch["image"].to(device))
            heat, size, off = model.split(o)
            dets = decode(heat, size, off, model.stride, k, thr)
            for idx, d in zip(batch["index"].tolist(), dets):
                out[idx] = d
    return out


def run_eval(model, ds, device, classes):
    pred = predict(model, ds, device)
    aps, m = evaluate(pred, ds.gt_boxes(), len(classes))
    return aps, m, pred


def cmd_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = Detector(config=args.config).to(device)
    ds, classes = dev_datasets(model.stride, args.scale, args.test_fold, args.val_fold, args.seed, args.aug)
    if args.limit:
        for d in ds.values():
            d.frames = d.frames[: args.limit]
            d.images = d.images[: args.limit]
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"config {args.config} scale {args.scale}: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{macs(model, ds['train'].h, ds['train'].w) / 1e6:.1f}M MACs/frame, stride {model.stride}, "
          f"train/val/test {len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])}, device {device}")
    dl = DataLoader(ds["train"], args.bs, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(dl), pct_start=0.1)
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), np.zeros(3)
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items() if k != "index"}
            heat, size, off = model.split(model(b["image"]))
            l_heat = focal_loss(heat, b["heat"])
            l_size = reg_l1(size, b["ind"], b["size"], b["mask"])
            l_off = reg_l1(off, b["ind"], b["off"], b["mask"])
            loss = l_heat + 0.1 * l_size + l_off
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += (l_heat.item(), l_size.item(), l_off.item())
        tot /= len(dl)
        aps, m, _ = run_eval(model, ds["val"], device, classes)
        log.append({"epoch": ep + 1, "loss_heat": tot[0], "loss_size": tot[1], "loss_off": tot[2], "val_map": m})
        flag = ""
        if m > best:
            best = m
            torch.save({"model": model.state_dict(), "config": args.config, "scale": args.scale,
                        "classes": classes, "epoch": ep + 1, "val_map": m}, out / "best.pt")
            flag = " *"
        print(f"ep {ep + 1:3d}  heat {tot[0]:.3f} size {tot[1]:.3f} off {tot[2]:.3f}  "
              f"val mAP {100 * m:5.1f}  ({time.time() - t0:.0f}s){flag}", flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    report(model, ds, device, classes, out, ck["epoch"])


def report(model, ds, device, classes, out, epoch):
    res = {"epoch": epoch}
    for split in ("val", "test"):
        aps, m, pred = run_eval(model, ds[split], device, classes)
        res[split] = {"map": m, "ap": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)}}
        print(f"{split}: mAP@0.5 {100 * m:.1f} (best epoch {epoch})")
        for c, a in zip(classes, aps):
            n = sum(sum(g[0] == i for g in gs) for gs in ds[split].gt_boxes().values() for i in [classes.index(c)])
            print(f"   {c:16s} {'  n/a' if np.isnan(a) else f'{100 * a:5.1f}'}  ({n} boxes)")
        (out / f"pred_{split}.json").write_text(json.dumps({str(k): v for k, v in pred.items()}))
    (out / "results.json").write_text(json.dumps(res, indent=1))


def cmd_quantize(args):
    """M2: calibrate, build the integer model, evaluate it, export weights + goldens."""
    from .quant import IntDetector, calibrate

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = Detector(config=ck["config"]).to(device).eval()
    model.load_state_dict(ck["model"])
    ds, classes = dev_datasets(model.stride, ck["scale"], args.test_fold, args.val_fold)
    out = (ROOT / args.checkpoint).parent

    def u8(d, idx):  # working-scale uint8 frames
        return np.stack([np.round(d._prep(d.images[i]) * 255).astype(np.uint8) for i in idx])

    rng = np.random.default_rng(0)
    calib = u8(ds["train"], rng.choice(len(ds["train"]), args.calib, replace=False))
    scales = calibrate(model, calib, device, args.pct)
    print("activation scales (x255 = uint8 range top):", [f"{s * 255:.2f}" for s in scales])
    q = IntDetector(model, scales)
    res = {"calib": args.calib, "pct": args.pct, "act_scales": scales}
    for split in ("val", "test"):
        d = ds[split]
        pred = {}
        for a in range(0, len(d), 8):
            idx = list(range(a, min(a + 8, len(d))))
            acc, _ = q.forward(u8(d, idx), device)
            heat, size, off = q.split_real(acc)
            for i, dets in zip(idx, decode(heat, size, off, model.stride, 32, -4.0)):
                pred[d.frames[i].index] = dets
        aps, m = evaluate(pred, d.gt_boxes(), len(classes))
        fl = json.loads((out / "results.json").read_text())[split]["map"]
        print(f"{split}: int8 mAP@0.5 {100 * m:.1f}  (float {100 * fl:.1f}, delta {100 * (m - fl):+.1f})")
        res[split] = {"map_int8": m, "map_float": fl, "ap": {c: (None if np.isnan(a) else float(a)) for c, a in zip(classes, aps)}}
    (out / "quant_results.json").write_text(json.dumps(res, indent=1))
    q.export(out / "int_model.json")
    # golden vectors: a few test frames with the exact int32 head map
    gold_idx = [0, len(ds["test"]) // 2, len(ds["test"]) - 1]
    imgs = u8(ds["test"], gold_idx)
    acc, _ = q.forward(imgs, device)
    np.savez_compressed(out / "golden.npz", images=imgs, head=acc.cpu().numpy().astype(np.int32),
                        frame_index=np.array([ds["test"].frames[i].index for i in gold_idx]))
    print(f"wrote {out / 'int_model.json'}, {out / 'golden.npz'}, {out / 'quant_results.json'}")


def cmd_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / args.checkpoint, map_location=device, weights_only=False)
    model = Detector(config=ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    ds, classes = dev_datasets(model.stride, ck["scale"], args.test_fold, args.val_fold)
    report(model, ds, device, classes, (ROOT / args.checkpoint).parent, ck["epoch"])


def main():
    p = argparse.ArgumentParser(prog="sonar-edge")
    sub = p.add_subparsers(required=True)

    def folds(sp):
        sp.add_argument("--test-fold", type=int, default=0)
        sp.add_argument("--val-fold", type=int, default=1)

    t = sub.add_parser("train")
    t.add_argument("--config", default="base")
    t.add_argument("--scale", type=int, default=1, help="input downsample factor")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--aug", choices=["basic", "strong"], default="basic")
    t.add_argument("--limit", type=int, default=0, help="use only N frames per split (smoke test)")
    t.add_argument("--out", default="runs/dev")
    folds(t)
    t.set_defaults(func=cmd_train)

    qz = sub.add_parser("quantize")
    qz.add_argument("checkpoint")
    qz.add_argument("--calib", type=int, default=64, help="calibration frames from the train split")
    qz.add_argument("--pct", type=float, default=99.99, help="activation percentile for the uint8 range")
    folds(qz)
    qz.set_defaults(func=cmd_quantize)

    e = sub.add_parser("eval")
    e.add_argument("checkpoint")
    folds(e)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)
