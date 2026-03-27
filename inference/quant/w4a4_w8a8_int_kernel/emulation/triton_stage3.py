"""
Triton kernels for Stage3 (4-to-1 reduction) in NVFP emulation.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@triton.jit
def _stage3_reduce4_kernel(
    x_ptr,
    max_exp_ptr,
    y_ptr,
    numel,
    W,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    base = offs * 4
    v0 = tl.load(x_ptr + base + 0, mask=mask, other=0.0).to(tl.float64)
    v1 = tl.load(x_ptr + base + 1, mask=mask, other=0.0).to(tl.float64)
    v2 = tl.load(x_ptr + base + 2, mask=mask, other=0.0).to(tl.float64)
    v3 = tl.load(x_ptr + base + 3, mask=mask, other=0.0).to(tl.float64)

    max_exp = tl.load(max_exp_ptr + offs, mask=mask, other=0).to(tl.float64)
    scale = tl.exp2(W - max_exp)

    s0 = v0 * scale
    s1 = v1 * scale
    s2 = v2 * scale
    s3 = v3 * scale

    t0 = tl.where(s0 >= 0, tl.floor(s0), tl.ceil(s0))
    t1 = tl.where(s1 >= 0, tl.floor(s1), tl.ceil(s1))
    t2 = tl.where(s2 >= 0, tl.floor(s2), tl.ceil(s2))
    t3 = tl.where(s3 >= 0, tl.floor(s3), tl.ceil(s3))

    out = (t0 + t1 + t2 + t3) / scale
    tl.store(y_ptr + offs, out, mask=mask)


def triton_stage3_reduce4(grouped: torch.Tensor, max_exp: torch.Tensor, W: int) -> torch.Tensor:
    """
    Args:
        grouped: [M, N, G4, 4], float32 CUDA contiguous tensor.
        max_exp: [M, N, G4], exponent from torch.frexp(max_abs(grouped, dim=-1)).
        W: stage3 bit width.

    Returns:
        Tensor [M, N, G4], float64.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not grouped.is_cuda or not max_exp.is_cuda:
        raise RuntimeError("Triton Stage3 kernel requires CUDA tensors")
    if grouped.dtype != torch.float32:
        grouped = grouped.float()
    if max_exp.dtype != torch.int32:
        max_exp = max_exp.to(torch.int32)
    if not grouped.is_contiguous():
        grouped = grouped.contiguous()
    if not max_exp.is_contiguous():
        max_exp = max_exp.contiguous()
    if grouped.shape[-1] != 4:
        raise ValueError(f"Expected grouped last dim == 4, got {grouped.shape}")
    if grouped.shape[:-1] != max_exp.shape:
        raise ValueError(f"Shape mismatch: grouped={grouped.shape}, max_exp={max_exp.shape}")

    m, n, g4, _ = grouped.shape
    x = grouped.view(-1, 4)
    exp_flat = max_exp.view(-1)
    y = torch.empty((m * n * g4,), device=grouped.device, dtype=torch.float64)
    numel = y.numel()

    BLOCK = 256
    grid = (triton.cdiv(numel, BLOCK),)
    _stage3_reduce4_kernel[grid](
        x_ptr=x,
        max_exp_ptr=exp_flat,
        y_ptr=y,
        numel=numel,
        W=float(W),
        BLOCK=BLOCK,
    )
    return y.view(m, n, g4)