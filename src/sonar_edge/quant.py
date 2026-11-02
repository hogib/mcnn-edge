"""M2: integer model. Folds BN, quantizes, and runs the network with exactly the
arithmetic the C runtime uses, so the two can be compared bit for bit.

Scheme
  activations  uint8, scale s_a, zero point 0 (every activation follows ReLU or
               is the raw pixel, so it is non-negative)
  weights      int8, symmetric, one scale per output channel
  accumulate   int32:  acc = sum(w * a) + bias_q   (bias_q = round(b / (s_w * s_in)))
  requantize   out = clip(round_shift(acc * M, shift), 0, 255)
               with M an int32 multiplier in [2^30, 2^31) and a right shift,
               i.e. s_w * s_in / s_out as a fixed-point number
  head         no requantization: int32 accumulators are returned with the
               per-channel scale s_w * s_in, and decoded directly.

Integer convolutions are evaluated in float64 on torch (CPU or GPU), which is
exact here: every product is < 2^15 and sums have < 2^12 terms, far below 2^53.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Detector


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Return float (W, b) with BN folded into the conv."""
    w = conv.weight.detach().cpu().numpy().astype(np.float64)
    g = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    mu = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    k = g / np.sqrt(var + bn.eps)
    return w * k[:, None, None, None], beta - mu * k


def quantize_weights(w):
    """Per-output-channel symmetric int8. Returns (w_q int8, scale per channel)."""
    amax = np.abs(w).reshape(w.shape[0], -1).max(1)
    scale = np.where(amax > 0, amax / 127.0, 1.0)
    w_q = np.clip(np.round(w / scale[:, None, None, None]), -127, 127).astype(np.int8)
    return w_q, scale


def fixed_point(x):
    """Real x in (0, 1) -> (M, shift) with M an int32 in [2^30, 2^31) and x ~= M * 2^-shift."""
    assert 0 < x < 1, x
    shift = 0
    while x < 0.5:
        x *= 2
        shift += 1
    M = int(round(x * (1 << 31)))
    if M == (1 << 31):
        M >>= 1
        shift -= 1
    return M, shift + 31


def round_shift(acc, M, shift):
    """(acc * M) >> shift with rounding half away from zero, as the C runtime does it
    (int64 product, then shift). acc, M, shift: int64 tensors (broadcastable)."""
    prod = acc * M
    half = torch.ones_like(prod) << (shift - 1)
    return torch.where(prod >= 0, (prod + half) >> shift, -((-prod + half) >> shift))


def conv2d_int(x, w_q, stride, dilation):
    """Exact integer convolution. x: [B,Cin,H,W] int64 tensor, w_q: int8 numpy
    [Cout,Cin,k,k]. Zero padding = dilation for 3x3, 0 for 1x1. Returns int64."""
    w = torch.from_numpy(w_q.astype(np.float64)).to(x.device)
    pad = dilation if w_q.shape[-1] == 3 else 0
    y = F.conv2d(x.to(torch.float64), w, stride=stride, padding=pad, dilation=dilation)
    return y.round().to(torch.int64)


class IntLayer:
    def __init__(self, w_q, bias_q, stride, dilation, M=None, shift=None, out_scale=None, w_scale=None):
        self.w_q, self.bias_q = w_q, bias_q
        self.stride, self.dilation = stride, dilation
        self.M, self.shift, self.out_scale, self.w_scale = M, shift, out_scale, w_scale


class IntDetector:
    """Integer inference. Built from a float Detector plus calibration ranges."""

    def __init__(self, model: Detector, act_scales):
        """act_scales: output scale per backbone layer (real value of uint8 step), from calibration."""
        self.stride = model.stride
        self.n_classes = model.n_classes
        self.layers = []
        s_in = 1.0 / 255.0  # raw pixel 0..255 represents 0..1
        for blk, s_out in zip(model.backbone, act_scales):
            conv, bn = blk[0], blk[1]
            w, b = fold_bn(conv, bn)
            w_q, s_w = quantize_weights(w)
            bias_q = np.round(b / (s_w * s_in)).astype(np.int64)
            Ms, shifts = zip(*(fixed_point(sw * s_in / s_out) for sw in s_w))
            self.layers.append(IntLayer(w_q, bias_q, conv.stride[0], conv.dilation[0],
                                        np.array(Ms, np.int64), np.array(shifts, np.int64), s_out, s_w))
            s_in = s_out
        head = model.head
        w = head.weight.detach().cpu().numpy().astype(np.float64)
        b = head.bias.detach().cpu().numpy().astype(np.float64)
        w_q, s_w = quantize_weights(w)
        self.head = IntLayer(w_q, np.round(b / (s_w * s_in)).astype(np.int64), 1, 1, w_scale=s_w)
        self.head_scale = s_w * s_in  # per output channel: real = acc * head_scale

    def forward(self, img_u8, device="cpu"):
        """img_u8: [B,H,W] uint8 numpy -> int64 head map [B, C+4, gh, gw] (int32 range)
        and its per-channel real scale. Every intermediate is exactly what the C runtime computes."""
        x = torch.from_numpy(img_u8.astype(np.int64))[:, None].to(device)
        t = lambda a: torch.from_numpy(np.asarray(a, np.int64)).to(device)
        for L in self.layers:
            acc = conv2d_int(x, L.w_q, L.stride, L.dilation) + t(L.bias_q)[None, :, None, None]
            assert acc.abs().max() < 2**31, "int32 overflow"
            y = round_shift(acc, t(L.M)[None, :, None, None], t(L.shift)[None, :, None, None])
            x = y.clamp(0, 255)
        acc = conv2d_int(x, self.head.w_q, 1, 1) + t(self.head.bias_q)[None, :, None, None]
        assert acc.abs().max() < 2**31, "int32 overflow in head"
        return acc, self.head_scale

    def split_real(self, acc):
        """int head map -> float (heat logits, size, offset), like the float model's split()."""
        real = acc.to(torch.float64) * torch.from_numpy(self.head_scale).to(acc.device)[None, :, None, None]
        real = real.to(torch.float32)
        C = self.n_classes
        return real[:, :C], real[:, C : C + 2], real[:, C + 2 : C + 4]

    def export(self, path: Path):
        """Weights and scales as JSON (consumed by the C header generator in M3)."""
        d = {
            "stride": self.stride, "n_classes": self.n_classes,
            "layers": [{
                "cout": int(L.w_q.shape[0]), "cin": int(L.w_q.shape[1]),
                "stride": L.stride, "dilation": L.dilation,
                "w": L.w_q.flatten().tolist(), "bias": L.bias_q.tolist(),
                "M": L.M.tolist(), "shift": L.shift.tolist(),
            } for L in self.layers],
            "head": {"cout": int(self.head.w_q.shape[0]), "cin": int(self.head.w_q.shape[1]),
                     "w": self.head.w_q[:, :, 0, 0].flatten().tolist(), "bias": self.head.bias_q.tolist(),
                     "scale": self.head_scale.tolist()},
        }
        path.write_text(json.dumps(d))


@torch.no_grad()
def calibrate(model: Detector, images, device, pct=99.99):
    """Per-layer activation scale: percentile of post-ReLU outputs over calibration frames / 255."""
    model.eval()
    maxes = [[] for _ in model.backbone]
    hooks = [blk.register_forward_hook(lambda m, i, o, k=k: maxes[k].append(o.detach().flatten().cpu().numpy()))
             for k, blk in enumerate(model.backbone)]
    for img in images:
        model(torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device))
    for h in hooks:
        h.remove()
    rng = np.random.default_rng(0)
    scales = []
    for m in maxes:
        v = np.concatenate([rng.choice(a, min(len(a), 200_000), replace=False) for a in m])
        scales.append(float(np.percentile(v, pct)) / 255.0)
    return scales
