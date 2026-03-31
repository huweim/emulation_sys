from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _get_vllm_ops():
    import vllm._custom_ops as vllm_ops

    return vllm_ops


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}."
    shape_excl_last = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape_excl_last

def sym_int8_quant(
    x: torch.Tensor,
    scales: torch.Tensor | None = None,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    vllm_ops = _get_vllm_ops()
    assert x.is_cuda, "x must be a CUDA tensor"
    x_2d, _ = _flatten_to_2d(x)
    if x_2d.dtype not in (torch.float16, torch.bfloat16):
        x_2d = x_2d.to(torch.float16)

    if scales is None:
        q, scales_2d, _ = vllm_ops.scaled_int8_quant(x_2d.contiguous())
        if clip_ratio != 1.0:
            scales_2d = scales_2d * clip_ratio
            q = torch.round(x_2d / scales_2d).clamp(-128, 127).to(torch.int8).contiguous()
    else:
        assert scales.is_cuda, "scales must be a CUDA tensor"
        scales_2d = scales.reshape(-1, 1).to(device=x_2d.device, dtype=torch.float32)
        q = torch.round(x_2d.to(torch.float32) / scales_2d).clamp(-128, 127).to(torch.int8).contiguous()
    return q, scales_2d.contiguous()

def int8_linear(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    vllm_ops = _get_vllm_ops()
    assert out_dtype in (torch.float16, torch.bfloat16), "vLLM cutlass_scaled_mm supports fp16/bf16 output"
    a_scale = a_scales.reshape(-1, 1).to(device=a_q.device, dtype=torch.float32).contiguous()
    b_scale = b_scales.reshape(-1, 1).to(device=b_q.device, dtype=torch.float32).contiguous()
    return vllm_ops.cutlass_scaled_mm(
        a_q.contiguous(),
        b_q.t(),
        a_scale,
        b_scale,
        out_dtype,
    )
