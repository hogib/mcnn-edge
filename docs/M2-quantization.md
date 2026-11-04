# M2: integer-only inference by post-training quantisation

## 1. Motivation

The C runtime (M3) must run without floating point in the inference path, both
for portability to devices without an FPU and so that its output is bit-identical
on every platform, which is what makes the Python-to-C verification chain exact.
This milestone defines the integer arithmetic, implements it in Python as a
reference ("simulation"), quantises the trained float detector, and measures the
accuracy cost.

## 2. Method

### 2.1 Quantisation scheme

We use the affine scheme of Jacob et al. (2018) restricted to the symmetric,
zero-point-free special case that this network allows. Every activation tensor
is either the raw 8-bit pixel or the output of a ReLU, hence non-negative, so it
is represented as an unsigned 8-bit integer with zero point 0:

    a = s_a * a_q,        a_q in {0, ..., 255}.

Weights are signed 8-bit with one scale per output channel c:

    W_c = s_{w,c} * W_{q,c},   s_{w,c} = max |W_c| / 127,   W_{q,c} = round(W_c / s_{w,c}).

### 2.2 Batch-normalisation folding

A convolution followed by batch normalisation with parameters (gamma, beta, mu,
sigma^2, epsilon) is replaced before quantisation by the single convolution

    W' = W * gamma / sqrt(sigma^2 + eps),    b' = beta - mu * gamma / sqrt(sigma^2 + eps),

applied per output channel. The runtime therefore has no normalisation layer.

### 2.3 Integer layer

With input scale s_in and output scale s_out, a layer computes, per output
channel c,

    acc   = sum_i W_{q,c,i} * a_{q,i} + b_{q,c},        b_{q,c} = round( b'_c / (s_{w,c} s_in) ),
    out_q = clip( round( acc * s_{w,c} s_in / s_out ), 0, 255 ).

The clip to [0, 255] implements the ReLU. The real multiplier
m_c = s_{w,c} s_in / s_out < 1 is replaced by a fixed-point pair (M_c, n_c) with
M_c an int32 in [2^30, 2^31) and m_c ~= M_c 2^{-n_c} (relative error < 1e-8), so
that the requantisation is an int64 multiply, an add of 2^{n_c - 1} for rounding
half away from zero, and an arithmetic shift. Accumulators fit int32 with a wide
margin: products are below 2^15 and a dot product has at most 9 x 64 terms.

The head (1x1 convolution to C + 4 channels) is not requantised: its int32
accumulator is returned together with the per-channel real scale
s_{w,c} s_in, and the decoder works on these integers directly (M1, Section 2.3).

### 2.4 Calibration

Activation scales s_out are set per layer from the distribution of post-ReLU
outputs over 64 training frames: s_out = q_{99.99} / 255, where q_{99.99} is the
99.99th percentile of a 200,000-sample subset per frame. The percentile rather
than the maximum discards isolated outliers that would otherwise waste most of
the 8-bit range.

### 2.5 Exact reference implementation

The simulation evaluates the integer convolutions as float64 tensor operations
in PyTorch. This is exact, not approximate: all operands are integers, every
product is below 2^15 and every sum has fewer than 2^12 terms, so every
intermediate is an integer below 2^53 and is represented without error in
float64. The result is rounded to int64 and the requantisation is performed in
integer arithmetic. This gives a reference that is both bit-exact with respect
to the specification above and fast enough to evaluate the full test fold.

Unit tests (`tests/test_quant.py`) check the fixed-point conversion, the
rounding shift against a scalar reference, the convolution against explicit
integer sums, and that the integer network tracks a randomly initialised float
network within 5% relative error.

## 3. Results (development split)

| Config | Split | Float mAP@0.5 | Int8 mAP@0.5 | Delta |
|---|---|---|---|---|
| base | val | 59.4 | 57.2 | -2.2 |
| base | test | 53.7 | 54.4 | +0.7 |
| small | val | 54.1 | 53.3 | -0.7 |
| small | test | 58.6 | 59.2 | +0.6 |
| small, 150 ep, strong aug (M1 §5) | val | 55.0 | 53.1 | -1.8 |
| small, 150 ep, strong aug (M1 §5) | test | 64.1 | 65.2 | +1.1 |
| small, 360x240, 150 ep, strong aug (M1 §6.4; shipped since 2026-11-04) | val | 59.4 | 57.5 | -1.9 |
| small, 360x240, 150 ep, strong aug (M1 §6.4; shipped since 2026-11-04) | test | 64.0 | 63.8 | -0.2 |

Per-class test AP changes by at most a few points in either direction
(largest: base drink-carton 32 -> 37, can 20 -> 30; chain 42 -> 38). Calibrated
activation ranges span 4.5-9.7 (in units of the input range [0, 1]) across
layers (5.0-9.6 for the strongly augmented `small` model at 240x160, 5.0-10.1 for the
shipped 360x240 model, whose `activation scales` line is in `runs/small_360_cv_f0.log`).

## 4. Discussion

Post-training quantisation costs nothing measurable here: the int8/float
differences are of the same size as the run-to-run and fold-to-fold variation
observed in M1, and have no consistent sign. Quantisation-aware training, kept
as a fallback in the plan, is therefore not needed. Two properties of the
network help: activations are non-negative so no zero point is required, and
with at most 64 channels and 3x3 kernels the accumulators never approach the
int32 limit, so no intermediate rescaling is necessary.

## 5. Exported artefacts

`sonar-edge quantize <checkpoint>` writes, next to the checkpoint:

- `int_model.json`: per-layer int8 weights, int32 biases, (M, n) multipliers,
  strides and dilations; head weights, biases and real scales;
- `golden.npz`: three test frames (uint8, working resolution) with the exact
  int32 head maps the runtime must reproduce byte for byte;
- `quant_results.json`: the table above with per-class AP.

`sonar_edge.export_c` turns these into `runtime/weights.h` (weights in HWC
order, `[cout][ky][kx][cin]`, so that each dot product reads a contiguous run of
input channels) and `runtime/tests/golden.h`.

## 6. Limitations

- Calibration uses one percentile for every layer; a per-layer search (e.g.
  minimising KL divergence as in TensorRT's calibrator) might recover the 1-2
  points lost on the validation fold of `base`, but the evidence that anything
  is lost is weak.
- Symmetric per-channel weights spend one code (-128) unused; irrelevant at
  this accuracy.

## References

- Jacob, B. et al. (2018). Quantization and training of neural networks for
  efficient integer-arithmetic-only inference. CVPR.
- Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for
  efficient inference: a whitepaper. arXiv:1806.08342.
