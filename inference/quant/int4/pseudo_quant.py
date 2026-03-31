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


def int4_pseudo_emulation_gemm(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Emulation path: emulate real int4 GEMM math in fp32 from precomputed int4 q.
    - y = (q_a @ q_b.T) * (a_scale * b_scale^T)
    """
    assert q_a.ndim == 2 and q_b.ndim == 2
    assert q_a.shape[1] == q_b.shape[1], "q_a and q_b must share K dimension"

    q_a_f = q_a.to(out_dtype)
    q_b_f = q_b.to(out_dtype)

    a_scale = a_scales.reshape(-1, 1).to(device=q_a_f.device, dtype=out_dtype)
    b_scale = b_scales.reshape(-1, 1).to(device=q_b_f.device, dtype=out_dtype)

    assert q_a_f.shape[0] == a_scale.shape[0], "q_a rows must match a_scales"
    assert q_b_f.shape[0] == b_scale.shape[0], "q_b rows must match b_scales"

    c_q = q_a_f @ q_b_f.t()
    scale_outer = a_scale @ b_scale.t()
    y = c_q * scale_outer
    return y.to(out_dtype)


def int4_unpack(packed_q: torch.Tensor, cols_src: int) -> torch.Tensor:
    """
    Unpack uint8-packed signed int4 values to int8 tensor with shape [rows, cols_src].
    """
    assert packed_q.ndim == 2, "packed_q must be [rows, cols_dst]"
    assert packed_q.dtype == torch.uint8, "packed_q must be torch.uint8"
    assert cols_src >= 0, "cols_src must be non-negative"

    low = (packed_q & 0x0F).to(torch.int16)
    high = ((packed_q >> 4) & 0x0F).to(torch.int16)
    pairs = torch.stack((low, high), dim=-1).reshape(packed_q.shape[0], -1)
    pairs = pairs[:, :cols_src]
    signed = torch.where(pairs >= 8, pairs - 16, pairs)
    return signed.to(torch.int8)
