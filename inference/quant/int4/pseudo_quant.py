import torch


def _flatten_to_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    assert x.ndim >= 1, f"x.ndim needs to be >= 1, but got {x.ndim}."
    shape_excl_last = x.shape[:-1]
    return x.reshape(-1, x.shape[-1]), shape_excl_last


def _compute_row_scales(x_2d: torch.Tensor, clip_ratio: float = 1.0) -> torch.Tensor:
    max_abs = x_2d.abs().amax(dim=-1, keepdim=True)
    scales = (max_abs / 7.0) * clip_ratio
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales


def int4_pseudo_quantize(
    x: torch.Tensor,
    scales: torch.Tensor | None = None,
    q: torch.Tensor | None = None,
    clip_ratio: float = 1.0,
    dequant_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Pseudo INT4 quantization: fp -> int4 rounding/clamp -> dequantized high precision.
    """
    assert x.ndim >= 1
    x_2d, shape_excl_last = _flatten_to_2d(x)
    x_f = x_2d.to(torch.float32)

    if scales is None:
        scale_2d = _compute_row_scales(x_f, clip_ratio=clip_ratio)
    else:
        scale_2d = scales.reshape(-1, 1).to(device=x.device, dtype=torch.float32)

    if q is None:
        q_f = torch.round(x_f / scale_2d)
        q_f = torch.clamp(q_f, -8, 7)
    else:
        # Use provided q (already computed elsewhere) to ensure identical inputs.
        q_f = q.reshape(x_2d.shape).to(device=x.device, dtype=torch.float32)
        q_f = torch.clamp(q_f, -8, 7)

    deq = q_f * scale_2d
    return deq.reshape(*shape_excl_last, x.shape[-1]).to(dequant_dtype)


def pseudo_quantize_linear_weight_to_int4(
    weight: torch.Tensor,
    clip_ratio: float = 1.0,
    dequant_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (dequantized_weight, per_row_scales) for emulation reference path.
    """
    w_2d = weight.reshape(-1, weight.shape[-1]).to(torch.float32)
    scales = _compute_row_scales(w_2d, clip_ratio=clip_ratio)
    w_deq = int4_pseudo_quantize(
        weight,
        scales=scales,
        clip_ratio=clip_ratio,
        dequant_dtype=dequant_dtype,
    )
    return w_deq, scales
