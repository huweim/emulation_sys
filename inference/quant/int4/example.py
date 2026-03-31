import torch

import int4.ops as ops
import int4.pseudo_quant as pseudo_quant


def main() -> None:
    torch.manual_seed(23)
    torch.cuda.manual_seed(23)
    torch.cuda.manual_seed_all(23)

    # A: activation [M, K], B: weight [N, K]
    m, n, k = 128, 128, 256
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(n, k, dtype=torch.float16, device="cuda")

    # -------- Real quant path --------
    a_q, a_s = ops.sym_int4_quant(a)
    b_q, b_s = ops.quantize_linear_weight_to_int4(b)
    out_real = ops.int4_packed_linear(a_q, b_q, a_s, b_s, out_dtype=torch.float16)

    print("[Real] output shape:", tuple(out_real.shape))
    print("[Real] output sample:\n", out_real[:3, :3])

    # -------- Pseudo quant path --------
    a_pseudo = pseudo_quant.int4_pseudo_quantize(a, dequant_dtype=torch.float32)
    b_pseudo = pseudo_quant.int4_pseudo_quantize(b, dequant_dtype=torch.float32)
    out_pseudo = (a_pseudo @ b_pseudo.t()).to(torch.float16)

    print("\n[Pseudo] output shape:", tuple(out_pseudo.shape))
    print("[Pseudo] output sample:\n", out_pseudo[:3, :3])

    # -------- Compare --------
    diff = out_real.float() - out_pseudo.float()
    abs_diff = diff.abs()

    print("\n=== Error stats (real - pseudo) ===")
    print("max abs:", abs_diff.max().item())
    print("mean abs:", abs_diff.mean().item())
    print("rmse:", torch.sqrt((diff * diff).mean()).item())

    # Row-wise top-k analysis
    row_max = abs_diff.max(dim=1).values
    topk = min(5, row_max.numel())
    vals, idx = torch.topk(row_max, k=topk)
    print(f"\nTop-{topk} rows by max abs error:")
    for i in range(topk):
        r = idx[i].item()
        print(f"  row {r:3d}: max_abs={vals[i].item():.6f}, mean_abs={abs_diff[r].mean().item():.6f}")

    # -------- Dequantized real weight (for emulation/layout use) --------
    b_deq = ops.dequantize_int4_weight_to_high_precision(b_q, b_s, dtype=torch.float32)
    print("\n[Dequant weight] shape:", tuple(b_deq.shape), "dtype:", b_deq.dtype)
    print("[Dequant weight] sample:\n", b_deq[:2, :8])


if __name__ == "__main__":
    main()
