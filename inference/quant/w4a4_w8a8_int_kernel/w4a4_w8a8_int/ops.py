import sys
from pathlib import Path

import torch

import w4a4_w8a8_int_ops


def quantize_w4a4_act(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize W4A4 activation:
      input  : fp16/bf16, shape [M, K]
      output : int8,     shape [M, K/2]
      scales : fp16/bf16, shape [K/64, M]
    """
    input = input.contiguous()
    return w4a4_w8a8_int_ops.quantize_w4a4_act(input)


def quantize_w4a4_wgt(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize W4A4 weight:
      weight : fp16/bf16, shape [N, K]
      output : int8,      shape [N, K/2]
      scales : fp16/bf16, shape [K/64, N]
    """
    weight = weight.contiguous()
    return w4a4_w8a8_int_ops.quantize_w4a4_wgt(weight)


def gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
    act_unsigned: bool = False,
    fuse_silu: bool = False,
) -> torch.Tensor:
    """
    W4A4 int4 GEMM (linear):
      act     : int8, shape [M, K/2]
      wgt     : int8, shape [N, K/2]
      ascales : fp16/bf16, shape [K/64, M]
      wscales : fp16/bf16, shape [K/64, N]
      bias    : fp16/bf16, optional shape [N]
    """
    act = act.contiguous()
    wgt = wgt.contiguous()
    ascales = ascales.contiguous()
    wscales = wscales.contiguous()
    if bias is None:
        return w4a4_w8a8_int_ops.gemm_w4a4(act, wgt, ascales, wscales, act_unsigned, fuse_silu)
    return w4a4_w8a8_int_ops.gemm_w4a4_bias(act, wgt, ascales, wscales, bias.contiguous(), act_unsigned, fuse_silu)


def quantize_w8a8_act(input: torch.Tensor, fuse_glu: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize W8A8 activation:
      input  : bf16, shape [M, K]
      output : int8, shape [M, K] (fuse_glu=False) or [M, K/2] (fuse_glu=True)
      scales : bf16, shape [M]
    """
    input = input.contiguous()
    return w4a4_w8a8_int_ops.quantize_w8a8_act(input, fuse_glu)


def quantize_w8a8_wgt(weight: torch.Tensor, bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize W8A8 weight using DeepCompressor Nunchaku backend packer:
      weight : bf16/fp16, shape [N, K]
      output : int8 packed, shape [N, K]
      scales : bf16/fp16 packed, shape [N]
      bias   : bf16/fp16 packed, shape [N]
    """
    weight = weight.contiguous()
    if weight.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"weight must be bf16/fp16, got {weight.dtype}")

    # per-output-channel absmax scale, avoid zero scale
    wmax = weight.abs().amax(dim=1)
    wscales = (wmax / 127.0).clamp_min(torch.finfo(weight.dtype).tiny).to(weight.dtype)

    if bias is not None:
        bias = bias.contiguous().to(weight.dtype)

    root = Path(__file__).resolve().parents[1]
    deepcompressor_root = root / "third_party" / "deepcompressor"
    if str(deepcompressor_root) not in sys.path:
        sys.path.append(str(deepcompressor_root))

    from deepcompressor.backend.nunchaku.utils import convert_to_nunchaku_w8x8y16_linear_weight

    qweight, packed_wscales, packed_bias = convert_to_nunchaku_w8x8y16_linear_weight(weight, wscales, bias)
    return qweight.contiguous(), packed_wscales.contiguous(), packed_bias.contiguous()


def gemm_w8a8(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    W8A8 int8 GEMM (linear):
      act     : int8, shape [M, K]
      wgt     : int8, shape [N, K]
      ascales : bf16, shape [M]
      wscales : bf16, shape [N]
      bias    : bf16, optional shape [N]
    """
    act = act.contiguous()
    wgt = wgt.contiguous()
    ascales = ascales.contiguous()
    wscales = wscales.contiguous()
    if bias is None:
        return w4a4_w8a8_int_ops.gemm_w8a8(act, wgt, ascales, wscales)
    return w4a4_w8a8_int_ops.gemm_w8a8_bias(act, wgt, ascales, wscales, bias.contiguous())

