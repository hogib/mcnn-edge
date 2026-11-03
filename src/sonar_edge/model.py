"""Single-stage anchor-free detector (CenterNet-style) for 320x480 sonar frames.

Plain conv stack, no residuals or upsampling, so the C runtime needs only:
conv3x3 (stride 1/2, dilation 1..8) + folded BN + ReLU, and a 1x1 head conv.

Output stride 8: a 480x320 frame gives a 60x40 grid. Heads per cell:
  [0:C]      class heatmap logits (C = n_classes)
  [C:C+2]    box size (w, h) in units of stride
  [C+2:C+4]  centre offset (dx, dy) within the cell, in [0, 1)
"""

import torch
import torch.nn as nn

# name -> list of (out_ch, stride, dilation); kernel is 3x3 throughout
CONFIGS = {
    # full-res 480x320 input, ~280M MACs; RF 255 px
    "base": [(16, 2, 1), (32, 2, 1), (48, 2, 1), (48, 1, 1), (48, 1, 2), (48, 1, 4), (48, 1, 8)],
    # wider dilated stack
    "wide": [(16, 2, 1), (32, 2, 1), (64, 2, 1), (64, 1, 1), (64, 1, 2), (64, 1, 4), (64, 1, 8)],
    # narrower, for the half-res input (stride 8 of a 240x160 frame -> 30x20 grid)
    "small": [(16, 2, 1), (32, 2, 1), (32, 2, 1), (32, 1, 1), (32, 1, 2), (32, 1, 4)],
    # small's half-res layout with wide's channel widths: separates the effect of width from resolution
    "small_wide": [(16, 2, 1), (32, 2, 1), (64, 2, 1), (64, 1, 1), (64, 1, 2), (64, 1, 4)],
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, stride, dilation):
        super().__init__(
            nn.Conv2d(cin, cout, 3, stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Detector(nn.Module):
    def __init__(self, n_classes=11, config="base"):
        super().__init__()
        self.n_classes = n_classes
        self.config = config
        layers, cin = [], 1
        self.stride = 1
        for cout, s, d in CONFIGS[config]:
            layers.append(ConvBNReLU(cin, cout, s, d))
            cin, self.stride = cout, self.stride * s
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(cin, n_classes + 4, 1)
        # CenterNet init: heatmap bias so initial p ~ 0.01, keeps focal loss stable
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:n_classes] = -2.19

    def forward(self, x):
        return self.head(self.backbone(x))

    def split(self, out):
        C = self.n_classes
        return out[:, :C], out[:, C : C + 2], out[:, C + 2 : C + 4]


def macs(model: Detector, h=480, w=320) -> int:
    """Multiply-accumulates for one frame."""
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            h = (h + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) // m.stride[0] + 1
            w = (w + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1
            total += h * w * m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1]
    return total


def receptive_field(config) -> int:
    rf, jump = 1, 1
    for _, s, d in CONFIGS[config]:
        rf += 2 * d * jump
        jump *= s
    return rf


if __name__ == "__main__":
    for name in CONFIGS:
        m = Detector(config=name)
        scale = 2 if name.startswith("small") else 1
        print(f"{name:6s} stride {m.stride}  params {sum(p.numel() for p in m.parameters()):8,d}  "
              f"MACs {macs(m, 480 // scale, 320 // scale) / 1e6:6.1f}M  RF {receptive_field(name) * scale} px (input /{scale})")
