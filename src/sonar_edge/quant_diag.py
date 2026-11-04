"""Per-layer float-vs-int8 error for fold checkpoints (M1 §6.4, item 3).
  uv run python -m sonar_edge.quant_diag small_360_cv 0 1 2 3 4   [--n 48]
Prints, per layer, the relative RMS error of the dequantised uint8 activations against the
float activations, the saturation rate (values at 255) and the fraction of float values above
the uint8 range; then the head-logit error at the float model's peaks and the median relative
size error there. Checkpoints come from runs/<run>_f<k>/ (best.pt + quant_results.json)."""
import argparse, json
import numpy as np, torch
from .model import Detector
from .dataset import dev_datasets
from .quant import IntDetector, conv2d_int, round_shift

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = 48

def analyse(run, k):
    ck = torch.load(f"runs/{run}_f{k}/best.pt", map_location=dev, weights_only=False)
    model = Detector(config=ck["config"], seg=ck.get("seg", False), seg_up=ck.get("seg_up", 1)).to(dev).eval(); model.load_state_dict(ck["model"])
    q = json.load(open(f"runs/{run}_f{k}/quant_results.json"))
    im = IntDetector(model, q["act_scales"])
    ds, _ = dev_datasets(model.stride, ck["scale"], k, (k + 1) % 5)
    d = ds["test"]; idx = np.random.default_rng(0).choice(len(d), N, replace=False)
    u8 = np.stack([np.round(d._prep(d.images[i]) * 255).astype(np.uint8) for i in idx])
    # float activations
    acts = []
    hooks = [blk.register_forward_hook(lambda m, i, o: acts.append(o.detach().double())) for blk in model.backbone]
    with torch.no_grad():
        out_f = model(torch.from_numpy(u8.astype(np.float32) / 255.0)[:, None].to(dev)).double()
    for h in hooks: h.remove()
    # int activations, layer by layer
    x = torch.from_numpy(u8.astype(np.int64))[:, None].to(dev)
    t = lambda a: torch.from_numpy(np.asarray(a, np.int64)).to(dev)
    rows = []
    with torch.no_grad():
        for L, a_f in zip(im.layers, acts):
            acc = conv2d_int(x, L.w_q, L.stride, L.dilation) + t(L.bias_q)[None, :, None, None]
            y = round_shift(acc, t(L.M)[None, :, None, None], t(L.shift)[None, :, None, None])
            x = y.clamp(0, 255)
            a_q = x.double() * L.out_scale
            err = (a_q - a_f).pow(2).mean().sqrt() / a_f.pow(2).mean().sqrt()
            sat = (x == 255).double().mean()
            clip = (a_f > 255 * L.out_scale).double().mean()  # float values above the uint8 range
            rows.append((float(err), float(sat), float(clip)))
        acc = conv2d_int(x, im.head.w_q, 1, 1) + t(im.head.bias_q)[None, :, None, None]
        out_q = acc.double() * torch.from_numpy(im.head_scale).to(dev)[None, :, None, None]
    C = model.n_classes
    hf, hq = out_f[:, :C], out_q[:, :C]
    # logit error at float peaks above threshold -4
    peak = (torch.nn.functional.max_pool2d(hf, 3, 1, 1) == hf) & (hf > -4)
    dl = (hq - hf)[peak]
    size_err = ((out_q[:, C:C+2] - out_f[:, C:C+2]).abs() / out_f[:, C:C+2].abs().clamp(min=1e-3))[peak[:, :1].expand(-1, 2, -1, -1)].median()
    return rows, float(dl.abs().mean()), float(dl.mean()), float(dl.abs().max()), int(peak.sum()), float(size_err)

def main():
    global N
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("folds", type=int, nargs="+"); ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args(); N = a.n
    for k in a.folds:
        rows, mae, bias, mx, n, se = analyse(a.run, k)
        print(f"{a.run:14s} f{k}  layer relRMS%: " + " ".join(f"{100*e:4.1f}" for e, _, _ in rows)
              + "  | sat%: " + " ".join(f"{100*s:4.2f}" for _, s, _ in rows)
              + f"  | peak logit |err| {mae:.3f} bias {bias:+.3f} max {mx:.2f} (n={n})  size rel err med {100*se:.1f}%")


if __name__ == "__main__":
    main()
