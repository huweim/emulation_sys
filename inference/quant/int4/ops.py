import importlib
import os
import sys
from functools import lru_cache
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _get_flatquant_deploy() -> ModuleType:
    """Lazily import FlatQuant deploy package."""
    flatquant_root = os.path.join(os.path.dirname(__file__), "third_party", "FlatQuant")
    if flatquant_root not in sys.path:
        sys.path.insert(0, flatquant_root)
    return importlib.import_module("deploy")


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}."
    shape_excl_last = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape_excl_last


def _compute_row_scales(x_2d: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    max_abs = x_2d.abs().amax(dim=-1, keepdim=True)
    scales = (max_abs / 7.0) * clip_ratio
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales.to(torch.float16)


def sym_int4_quant(
    x: torch.Tensor,
    scales: torch.Tensor | None = None,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    deploy = _get_flatquant_deploy()

    assert x.is_cuda, "x must be a CUDA tensor"
    x_2d, _ = _flatten_to_2d(x)
    if x_2d.dtype != torch.float16:
        x_2d = x_2d.to(torch.float16)

    if scales is None:
        scales_2d = _compute_row_scales(x_2d, clip_ratio=clip_ratio)
    else:
        assert scales.is_cuda, "scales must be a CUDA tensor"
        scales_2d = scales.reshape(-1, 1).to(device=x_2d.device, dtype=torch.float16)

    packed_int4 = deploy.sym_quant(x_2d.contiguous(), scales_2d.contiguous())
    return packed_int4, scales_2d


def int4_packed_gemm(a_packed: torch.Tensor, b_packed: torch.Tensor) -> torch.Tensor:
    deploy = _get_flatquant_deploy()
    assert a_packed.is_cuda and b_packed.is_cuda
    assert a_packed.dtype == torch.uint8 and b_packed.dtype == torch.uint8
    return deploy.matmul(a_packed.contiguous(), b_packed.contiguous())


def int4_packed_linear(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    deploy = _get_flatquant_deploy()
    int32_out = int4_packed_gemm(a_packed, b_packed)
    out = deploy.sym_dequant(
        int32_out,
        a_scales.reshape(-1, 1).contiguous().to(torch.float16),
        b_scales.reshape(-1, 1).contiguous().to(torch.float16),
        32,
    )
    return out.to(out_dtype)


def quantize_linear_weight_to_int4(
    weight: torch.Tensor,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sym_int4_quant(weight, scales=None, clip_ratio=clip_ratio)


def dequantize_int4_weight_to_high_precision(
    weight_int4_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    deploy = _get_flatquant_deploy()
    q_int = deploy.functional.unpack_i4(weight_int4_packed.contiguous()).to(torch.float32)
    scale = weight_scales.reshape(-1, 1).to(device=q_int.device, dtype=torch.float32)
    return (q_int * scale).to(dtype)
