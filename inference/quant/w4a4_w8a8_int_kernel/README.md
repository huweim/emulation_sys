# w4a4_w8a8_int_kernel

This directory packages **Nunchaku** W4A4 (int4 packed and stored as int8) and W8A8 (int8) CUDA kernels into a standalone, installable PyTorch CUDA extension.

## Installation

It is recommended to run the following steps from the repository root.

### 1) Pull submodules

```bash
git submodule update --init --recursive
```

### 2) Install `third_party/deepcompressor` first

There is a `pyav` dependency line in `deepcompressor`'s `pyproject.toml` that can cause installation errors. Remove that line before installation (removing it does not affect usage here).

First, enter the directory:

```bash
cd inference/quant/w4a4_w8a8_int_kernel/third_party/deepcompressor
```

Remove the `pyav` dependency line (command-line approach):

```bash
sed -i '/pyav/d' pyproject.toml
```

Install:

```bash
pip install .
```

### 3) Install `w4a4_w8a8_int_kernel`

```bash
cd ../../
pip install .
```

---

## Python API

After installation:

```python
import w4a4_w8a8_int as k
```

### W4A4

```python
# activation quant
act_q, act_scales = k.quantize_w4a4_act(x_fp16_or_bf16)   # x: [M, K] -> act_q: [M, K/2], scales: [K/64, M]

# weight quant
w_q, w_scales = k.quantize_w4a4_wgt(w_fp16_or_bf16)      # w: [N, K] -> w_q: [N, K/2], scales: [K/64, N]

# GEMM
out = k.gemm_w4a4(act_q, w_q, act_scales, w_scales, bias=None, act_unsigned=False, fuse_silu=False)  # out: [M, N]
out = k.gemm_w4a4(act_q, w_q, act_scales, w_scales, bias=b_fp16_or_bf16)  # with bias: b: [N]
```

Constraints (checked in the C++ wrapper):
- `K` must be a multiple of `64`
- In `gemm_w4a4`, `M` must be a multiple of `256`, and `N` must be a multiple of `128`
- During quantization, `M` must be a multiple of `32`

### W8A8

```python
# activation quant (BF16 only)
act_q, act_scales = k.quantize_w8a8_act(x_bf16, fuse_glu=False)  # x: [M, K] -> act_q: [M, K], scales: [M]

# weight quant
w_q, w_scales = k.quantize_w8a8_wgt(w_fp16_or_bf16)  # w: [N, K] -> w_q: [N, K], scales: [N]

# GEMM
out = k.gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=None)   # out: BF16 [M, N]
out = k.gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=b_bf16)
```

Constraints:
- The input dtype of `quantize_w8a8_act` must be BF16
- In `gemm_w8a8`, `M` must be a multiple of `256`, and `N` must be a multiple of `128`

---

## Example

An example script is provided in this directory: `example.py`

```bash
cd inference/quant/w4a4_w8a8_int_kernel
python example.py
```
