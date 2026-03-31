from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from inference.quant.int8 import ops, pseudo_quant  # type: ignore
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from inference.quant.int8 import ops, pseudo_quant  # type: ignore


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this INT8 example.")

    torch.manual_seed(23)
    torch.cuda.manual_seed_all(23)

    m, n, k = 128, 128, 256
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")

    a_q, a_s = ops.sym_int8_quant(a)
    b_q, b_s = ops.sym_int8_quant(b)
    out_real = ops.int8_linear(a_q, b_q, a_s, b_s, out_dtype=torch.float16)
    print("[Real] output shape:", tuple(out_real.shape))
    print("[Real] output sample:\n", out_real[:3, :3])

    a_pseudo = pseudo_quant.int8_pseudo_quantize(a, scales=a_s, dequant_dtype=torch.float32)
    b_pseudo = pseudo_quant.int8_pseudo_quantize(b, scales=b_s, dequant_dtype=torch.float32)
    out_pseudo = (a_pseudo @ b_pseudo.t()).to(torch.float16)
    print("\n[Pseudo] output shape:", tuple(out_pseudo.shape))
    print("[Pseudo] output sample:\n", out_pseudo[:3, :3])

    out_emul = pseudo_quant.int8_pseudo_emulation_gemm(
        a_q, b_q, a_s, b_s, out_dtype=torch.float16
    )
    print("\n[Emul GEMM] output shape:", tuple(out_emul.shape))
    print("[Emul GEMM] output sample:\n", out_emul[:3, :3])

    diff = out_real.float() - out_pseudo.float()
    abs_diff = diff.abs()
    print("\n=== Error stats (real - pseudo) ===")
    print("max abs:", abs_diff.max().item())
    print("mean abs:", abs_diff.mean().item())
    print("rmse:", torch.sqrt((diff * diff).mean()).item())

    diff_emul = out_real.float() - out_emul.float()
    abs_diff_emul = diff_emul.abs()
    print("\n=== Error stats (real - emul) ===")
    print("max abs:", abs_diff_emul.max().item())
    print("mean abs:", abs_diff_emul.mean().item())
    print("rmse:", torch.sqrt((diff_emul * diff_emul).mean()).item())


if __name__ == "__main__":
    main()
