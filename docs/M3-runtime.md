# M3: a dependency-free integer C runtime

## 1. Motivation

The deployment target is a Raspberry Pi, but the design goal is stricter than
"runs on a Pi": the inference path must have no dependencies beyond the C
standard library, no dynamic allocation, no floating point, and must produce
output that is bit-identical to the Python reference of M2 on any platform.
This rules out libtorch, ONNX Runtime and TensorFlow Lite, which would also
dominate any measurement of what the model itself costs, and it makes the
runtime portable to anything with a C11 compiler, including microcontrollers.

## 2. Design

### 2.1 Compiled-in model

`sonar_edge.export_c` renders the M2 export as `runtime/weights.h`: int8
kernels, int32 biases, per-channel fixed-point multipliers (M, n) and a layer
table of (c_in, c_out, stride, dilation), together with the input and output
geometry as preprocessor constants. Every tensor shape is therefore a
compile-time constant, all working memory is static, and the runtime is built
with `cc runtime/*.c`. The weights occupy 42 KB (`small`, at any input size) or 100 KB (`base`) of
read-only data.

### 2.2 Memory layout

Activations are stored HWC (channels innermost) in two static buffers that
alternate as input and output. Each tensor is stored with a zero border of
width equal to the *consuming* layer's dilation d, so that a 3x3 convolution
with stride s reads, for output (oy, ox) and tap (ky, kx), the padded pixel
(oy s + ky d, ox s + kx d) without any bounds check, exactly reproducing
PyTorch's `padding = dilation`. A layer writes its output already padded for
the next layer; the buffer is zeroed before each layer so borders are zero. The
buffer size is the largest padded tensor in the network, computed by the
exporter (160 KB for `small` at 240x160, 355 KB at the shipped 360x240, 606 KB for
`base`); total static memory is twice that plus the int32 head map (36 KB / 81 KB /
144 KB).

With channels innermost, each kernel tap is a dot product over a contiguous
run of c_in bytes against a contiguous run of c_in weights (`[c_out][ky][kx][c_in]`
order), which is the access pattern a NEON `udot`/`smlal` inner loop wants.

### 2.3 Arithmetic

Per output pixel, int32 accumulators (one per output channel) are initialised
with the bias and accumulated over the nine taps as
`acc += (int32) w * (int32) a` with w in [-127, 127] and a in [0, 255]. The
bound |acc| < 9 x 64 x 127 x 255 + |bias| < 2^28 leaves a margin of 8x to the
int32 limit. Requantisation is

    r = (acc * M + 2^(n-1)) >> n            (acc >= 0)
    r = -((-acc * M + 2^(n-1)) >> n)        (acc < 0)
    out = clamp(r, 0, 255)

computed in int64 (|acc M| < 2^28 x 2^31 = 2^59). Only non-negative quantities
are shifted, so the implementation-defined behaviour of right-shifting a
negative integer in C is never invoked. The clamp is the ReLU. The 1x1 head is
accumulated the same way and left in int32.

### 2.4 Decoder

For each class the int32 logit map is scanned for cells that are >= all of
their (existing) 3x3 neighbours, which matches `F.max_pool2d(x, 3, 1, 1) == x`
with its implicit -inf padding. Scores are compared with the threshold after
conversion to a real value in the same operation order as the Python
reference, `float32(float64(acc) * scale_c)`, and boxes are formed in double
precision with the same expression order as `targets.decode`. These are the
only floating-point operations in the runtime; they are per-detection scalars
with no accumulation, and the build uses `-std=c11`, under which GCC disables
floating-point contraction, so results are reproducible across platforms.
Detections are inserted into a fixed-size array sorted by score.

### 2.5 Interface

```c
int se_detect(const uint8_t *frame, float thr, se_detection_t *out, int max_out);
const int32_t *se_forward(const uint8_t *frame);   /* int32 head map, HWC */
```

Static buffers make the runtime non-reentrant; a Pi runs one detector.

## 3. Verification

`runtime/tests/test_golden.c` runs `se_forward` on the three test frames
exported by M2 and compares the entire int32 head map (30x20x15, 45x30x15 or 60x40x15
values) with the Python simulation. For both `small` and `base` the maps are
identical to the last bit on every cell, i.e. the C arithmetic implements the
specification of M2 exactly, including BN folding, per-channel scales,
rounding and saturation. The end-to-end check on the full test fold is M4.

## 4. Cost on a laptop CPU

Scalar C, `-O2`, one core of an AMD Ryzen 5 5600H, mean over >= 600 runs,
measured inside `sonar-detect -n` around `se_detect` (network + decoder):

| Config | Input | MACs / frame | Time / frame | Effective throughput |
|---|---|---|---|---|
| small | 240x160 | 34.8 M | 10.0 ms | 3.5 GMAC/s |
| small (shipped since 2026-11-04) | 360x240 | 78.4 M | 20.1 ms | 3.9 GMAC/s |
| base | 480x320 | 283.7 M | 71.0 ms | 4.0 GMAC/s |

About 2.5 multiply-accumulates per cycle at 4.4 GHz without vectorisation.
The Pi 4's Cortex-A72 at 1.5 GHz has lower clock and IPC; an estimate of 3-5x
these times (35-50 ms for `small` at 240x160, 70-100 ms at 360x240, 250-350 ms for
`base`) is to be verified in
M5, where an optional NEON inner loop is the obvious lever.

## 5. Build

GNU make, no configuration: `make` builds the static library and the CLI,
`make test` the golden test. Cross-compilation for a 64-bit Pi is
`make CC=aarch64-linux-gnu-gcc`; building on the Pi itself needs only gcc and
make. Flags: `-std=c11 -O2 -Wall -Wextra -Werror -pedantic`.

## 6. Limitations

- The scalar inner loop is the reference, not an optimised kernel; no SIMD yet.
- Each layer zeroes the whole output buffer rather than only its border; this
  costs under 1% of the convolution time and keeps the code simple.
- Frame decoding (PNG) is outside the runtime: the CLI reads binary PGM, which
  the Python side exports; a real sonar feed provides raw frames.
