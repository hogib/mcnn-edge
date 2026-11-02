import numpy as np
import torch

from sonar_edge.model import Detector
from sonar_edge.quant import IntDetector, conv2d_int, fixed_point, fold_bn, round_shift


def test_fixed_point_accuracy():
    for x in (0.9, 0.5, 0.123, 1e-3, 3.7e-6):
        M, shift = fixed_point(x)
        assert 2**30 <= M < 2**31
        assert abs(M * 2.0**-shift - x) / x < 1e-8


def test_round_shift_matches_reference():
    rng = np.random.default_rng(0)
    acc = torch.from_numpy(rng.integers(-2**31, 2**31, size=1000))
    M, shift = fixed_point(0.3)
    got = round_shift(acc, torch.tensor(M), torch.tensor(shift))
    ref = torch.from_numpy(np.array([int(np.floor(abs(a) * M / 2**shift + 0.5)) * (1 if a >= 0 else -1) for a in acc.tolist()]))
    assert torch.equal(got, ref)


def test_conv2d_int_is_exact():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.integers(0, 256, size=(2, 16, 40, 30)))
    w = rng.integers(-127, 128, size=(32, 16, 3, 3)).astype(np.int8)
    y = conv2d_int(x, w, stride=2, dilation=2)
    # reference with int64 arithmetic for a few output positions
    xp = torch.nn.functional.pad(x, (2, 2, 2, 2))
    for (b, co, oy, ox) in [(0, 0, 0, 0), (1, 31, 9, 7), (0, 5, 3, 14)]:
        ref = 0
        for ci in range(16):
            for ky in range(3):
                for kx in range(3):
                    ref += int(w[co, ci, ky, kx]) * int(xp[b, ci, oy * 2 + ky * 2, ox * 2 + kx * 2])
        assert int(y[b, co, oy, ox]) == ref


def test_int_detector_tracks_float_model():
    """Random (BN-folded) model: integer output must be close to the float output."""
    torch.manual_seed(0)
    m = Detector(config="small").eval()
    for blk in m.backbone:  # give BN non-trivial stats
        bn = blk[1]
        bn.running_mean.uniform_(-0.2, 0.2)
        bn.running_var.uniform_(0.5, 2.0)
        bn.weight.data.uniform_(0.5, 1.5)
        bn.bias.data.uniform_(-0.1, 0.3)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.2)
    img = (torch.rand(2, 1, 240, 160) * 255).round()
    with torch.no_grad():
        ref = m(img / 255.0)
        # calibration scales from the float activations themselves
        scales, x = [], img / 255.0
        for blk in m.backbone:
            x = blk(x)
            scales.append(float(x.max()) / 255.0)
    q = IntDetector(m, scales)
    acc, _ = q.forward(img[:, 0].numpy().astype(np.uint8))
    heat, size, off = q.split_real(acc)
    out = torch.cat([heat, size, off], 1)
    err = (out - ref).abs().max() / ref.abs().max()
    assert err < 0.05, float(err)
