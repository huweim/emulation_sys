from functools import lru_cache
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def _get_int4_cuda() -> ModuleType:
    from .cuda._build import load_int4_kernels

    mod = load_int4_kernels()
    assert mod is not None
    return mod


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}."
    shape_excl_last = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape_excl_last


def _compute_row_scales(x_2d: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    max_abs = x_2d.abs().amax(dim=-1, keepdim=True)
    scales = (max_abs / 7.0) * clip_ratio
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    # Keep scale dtype aligned with x dtype (fp16/bf16).
    return scales.to(dtype=x_2d.dtype)


def sym_int4_quant(
    x: torch.Tensor,
    scales: torch.Tensor | None = None,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    cuda = _get_int4_cuda()

    assert x.is_cuda, "x must be a CUDA tensor"
    x_2d, _ = _flatten_to_2d(x)
    if x_2d.dtype not in (torch.float16, torch.bfloat16):
        x_2d = x_2d.to(torch.float16)

    if scales is None:
        scales_2d = _compute_row_scales(x_2d, clip_ratio=clip_ratio)
    else:
        assert scales.is_cuda, "scales must be a CUDA tensor"
        scales_2d = scales.reshape(-1, 1).to(device=x_2d.device, dtype=x_2d.dtype)

    packed_int4 = cuda.sym_quant(x_2d.contiguous(), scales_2d.contiguous())
    return packed_int4, scales_2d


def int4_packed_gemm(a_packed: torch.Tensor, b_packed: torch.Tensor) -> torch.Tensor:
    cuda = _get_int4_cuda()
    assert a_packed.is_cuda and b_packed.is_cuda
    assert a_packed.dtype == torch.uint8 and b_packed.dtype == torch.uint8
    return cuda.matmul(a_packed.contiguous(), b_packed.contiguous())


def int4_packed_linear(
    a_packed: torch.Tensor,
    b_packed: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    int32_out = int4_packed_gemm(a_packed, b_packed)
    cuda = _get_int4_cuda()
    scale_dtype = torch.float16 if out_dtype == torch.float16 else torch.bfloat16
    out = cuda.sym_dequant(
        int32_out,
        a_scales.reshape(-1, 1).contiguous().to(dtype=scale_dtype),
        b_scales.reshape(-1, 1).contiguous().to(dtype=scale_dtype),
        32,
    )
    return out.to(out_dtype)
