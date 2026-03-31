# INT4 Kernels (Standalone)

This directory provides a standalone INT4 path for:

- symmetric quantization (`sym_quant`)
- CUTLASS INT4 GEMM (`matmul`)
- dequantization from GEMM `int32` output (`sym_dequant`)

The CUDA kernels are now independent from `FlatQuant/deploy` source files.

## Prerequisites

- Linux + NVIDIA GPU
- CUDA toolkit available (`nvcc` in `PATH`)
- Python 3.10+ recommended
- PyTorch with CUDA support

## Install

From repo root (`/state/partition/zhzhang/emulation_sys`):

```bash
# 1) pull submodules (includes int4 CUTLASS submodule)
git submodule update --init --recursive

# 2) install runtime deps in your environment
pip install torch
```

No separate wheel/build step is required for this INT4 module.

## How build works

`inference/quant/int4/cuda/_build.py` uses `torch.utils.cpp_extension.load()` and builds on first import/call.

First-time compile is triggered by calling any API in `inference.quant.int4.ops`, for example:

```python
import torch
from inference.quant.int4.ops import sym_int4_quant, int4_packed_gemm

x = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(256, 4096, device="cuda", dtype=torch.float16)

x_q, x_s = sym_int4_quant(x)
w_q, w_s = sym_int4_quant(w)
y_i32 = int4_packed_gemm(x_q, w_q)  # [128, 256], int32
```

## Optional build env vars

- `INT4_CUDA_ARCH`: override arch list, example `INT4_CUDA_ARCH="80;86;90"`
- `INT4_BUILD_VERBOSE=1`: enable verbose extension build logs

Example:

```bash
INT4_CUDA_ARCH="90" INT4_BUILD_VERBOSE=1 python -c "from inference.quant.int4.ops import sym_int4_quant"
```

## Test (real INT4 vs pseudo)

From repo root:

```bash
python -u -m inference.quant.int4.test_refactor
```

Optional env vars:

- `INT4_TEST_ITERS`: number of random test cases (default: `200`)
- `INT4_TEST_OUT_DTYPE`: `float16` or `bfloat16` (default: `float16`)
- `INT4_TEST_CLIP_RATIO`: clip ratio used by quantization (default: `1.0`)

Example:

```bash
INT4_TEST_ITERS=1000 INT4_TEST_OUT_DTYPE=float16 python -u -m inference.quant.int4.test_refactor
```

