"""
Triton kernels for Stage4 (FP32 accumulator + W-bit value accumulation) in NVFP emulation.
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
def _stage4_add_wbits_kernel(
    acc_ptr,
    new_ptr,
    scale_ptr,
    out_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    acc = tl.load(acc_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    new_val = tl.load(new_ptr + offs, mask=mask, other=0.0).to(tl.float64)
    scale = tl.load(scale_ptr + offs, mask=mask, other=1.0).to(tl.float64)

    acc_aligned = acc * scale
    new_scaled = new_val * scale
    new_trunc = tl.where(new_scaled >= 0, tl.floor(new_scaled), tl.ceil(new_scaled))

    out = (acc_aligned + new_trunc) / scale
    tl.store(out_ptr + offs, out, mask=mask)


def triton_stage4_add_wbits(acc_fp32: torch.Tensor, new_val_wbits: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        acc_fp32: FP32 accumulator tensor.
        new_val_wbits: W-bit emulated value tensor (float64 in practice).
        scale: per-element scale computed in torch via 2**(W - acc_exp).

    Returns:
        Tensor float64 with Stage4 fixed-point add result before final float32 rounding.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not acc_fp32.is_cuda or not new_val_wbits.is_cuda or not scale.is_cuda:
        raise RuntimeError("Triton Stage4 kernel requires CUDA tensors")
    if acc_fp32.shape != new_val_wbits.shape or acc_fp32.shape != scale.shape:
        raise ValueError(
            f"Shape mismatch: acc={acc_fp32.shape}, new={new_val_wbits.shape}, scale={scale.shape}"
        )

    acc = acc_fp32.contiguous()
    new_val = new_val_wbits.contiguous()
    scale_c = scale.contiguous()

    out = torch.empty_like(scale_c, dtype=torch.float64)
    numel = out.numel()

    BLOCK = 256
    grid = (triton.cdiv(numel, BLOCK),)
    _stage4_add_wbits_kernel[grid](
        acc_ptr=acc,
        new_ptr=new_val,
        scale_ptr=scale_c,
        out_ptr=out,
        numel=numel,
        BLOCK=BLOCK,
    )
    return out
