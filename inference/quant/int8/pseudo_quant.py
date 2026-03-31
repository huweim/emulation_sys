from __future__ import annotations

import torch


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}."
    shape_excl_last = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape_excl_last


def _compute_row_scales(x_2d: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    max_abs = x_2d.abs().amax(dim=-1, keepdim=True)
    scales = (max_abs / 127.0) * clip_ratio
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales


def int8_pseudo_quantize(
    x: torch.Tensor,
    scales: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    clip_ratio: float = 1.0,
    dequant_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert x.ndim >= 1
    x_2d, shape_excl_last = _flatten_to_2d(x)
    x_f = x_2d.to(torch.float32)

    if scales is None:
        scale_2d = _compute_row_scales(x_f, clip_ratio=clip_ratio)
    else:
        scale_2d = scales.reshape(-1, 1).to(device=x.device, dtype=torch.float32)

    if q is None:
        q_f = torch.round(x_f / scale_2d)
        q_f = torch.clamp(q_f, -128, 127)
    else:
        q_f = q.reshape(x_2d.shape).to(device=x.device, dtype=torch.float32)
        q_f = torch.clamp(q_f, -128, 127)

    deq = q_f * scale_2d
    return deq.reshape(*shape_excl_last, x.shape[-1]).to(dequant_dtype)

def int8_pseudo_emulation_gemm(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert q_a.ndim == 2 and q_b.ndim == 2
    assert q_a.shape[1] == q_b.shape[1], "q_a and q_b must share K dimension"

    a_scale = a_scales.reshape(-1, 1).to(device=q_a.device, dtype=torch.float32)
    b_scale = b_scales.reshape(-1, 1).to(device=q_b.device, dtype=torch.float32)

    # Match CUTLASS epilogue order:
    #   1) (accum int32 -> fp32) * scale_b (row / scalar)
    #   2) result * scale_a (col / scalar)
    #   3) cast to output dtype
    c = q_a.to(torch.float32) @ q_b.to(torch.float32).t()
    c = c * b_scale.t()
    c = c * a_scale
    return c.to(out_dtype)
