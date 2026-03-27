"""
只测试 quant/dequant（不做 GEMM），用于定位反量化问题来自 activation 还是 weight。

支持两种模式：
- w4a4
- w8a8

用法示例：
  python test_quant_dequant_only.py
  python test_quant_dequant_only.py --mode w4a4 --m 256 --n 128 --k 256 --dtype fp16 --seed 123
  python test_quant_dequant_only.py --mode w8a8 --m 256 --n 128 --k 256 --dtype bf16 --seed 123
"""
from __future__ import annotations

import argparse

import torch
import w4a4_w8a8_int as ops

import emulation.core as emu_core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["w4a4", "w8a8"], default="w4a4")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def error_stats(x_ref: torch.Tensor, x_hat: torch.Tensor) -> dict:
    diff = (x_ref.float() - x_hat.float()).abs()
    ref = x_ref.float().abs().clamp_min(1e-12)
    rel = diff / ref
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def print_stats(name: str, s: dict):
    print(
        f"[{name}] "
        f"max_abs={s['max_abs']:.6f}, mean_abs={s['mean_abs']:.6f}, "
        f"max_rel={s['max_rel']:.6f}, mean_rel={s['mean_rel']:.6f}"
    )


def _run_w4a4(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if K % 64 != 0:
        raise ValueError("K 必须是 64 的倍数（w4a4 量化要求）")
    if M % 32 != 0:
        raise ValueError("M 必须是 32 的倍数（quantize_w4a4_act 要求）")
    if N % 128 != 0:
        raise ValueError("N 必须是 128 的倍数（quantize_w4a4_wgt 要求）")

    act_fp = torch.randn(M, K, device="cuda", dtype=dtype)
    wgt_fp = torch.randn(N, K, device="cuda", dtype=dtype)

    act_q, act_scales = ops.quantize_w4a4_act(act_fp)
    wgt_q, wgt_scales = ops.quantize_w4a4_wgt(wgt_fp)

    act_int4 = emu_core.IntKernelUtils.unpack_int4_pairs_act(act_q, K).to(torch.float32)  # [M, K]
    wgt_int4 = emu_core.IntKernelUtils.unpack_int4_pairs_wgt(wgt_q, K).to(torch.float32)  # [N, K]

    G = K // 64
    act_deq = (act_int4.view(M, G, 64) * act_scales.t().float().unsqueeze(-1)).reshape(M, K).to(dtype)
    wgt_deq = (wgt_int4.view(N, G, 64) * wgt_scales.t().float().unsqueeze(-1)).reshape(N, K).to(dtype)

    return act_fp, act_deq, wgt_fp, wgt_deq


def _run_w8a8(M: int, N: int, K: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype != torch.bfloat16:
        raise ValueError("w8a8 模式下 activation 量化输入必须是 bf16，请使用 --dtype bf16")
    if M % 256 != 0:
        raise ValueError("M 必须是 256 的倍数（quantize/gemm_w8a8 约束）")
    if N % 128 != 0:
        raise ValueError("N 必须是 128 的倍数（quantize/gemm_w8a8 约束）")

    act_fp = torch.randn(M, K, device="cuda", dtype=dtype)
    wgt_fp = torch.randn(N, K, device="cuda", dtype=dtype)

    act_q, act_scales = ops.quantize_w8a8_act(act_fp, fuse_glu=False)
    wgt_q, wgt_scales, _ = ops.quantize_w8a8_wgt(wgt_fp)

    # w8a8 的权重默认是 nunchaku packed layout，需要先还原到逻辑 [N, K]
    wgt_int8 = emu_core.IntKernelUtils.unpack_nunchaku_w8_weight(wgt_q, N, K).to(torch.float32)
    act_int8 = act_q.to(torch.float32)

    act_deq = (act_int8 * act_scales.float().view(M, 1)).to(dtype)
    wgt_deq = (wgt_int8 * wgt_scales.float().view(N, 1)).to(dtype)

    return act_fp, act_deq, wgt_fp, wgt_deq


def main() -> int:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    M, N, K = args.m, args.n, args.k
    print(f"mode={args.mode}, seed={args.seed}, dtype={dtype}, M={M}, N={N}, K={K}")

    if args.mode == "w4a4":
        act_fp, act_deq, wgt_fp, wgt_deq = _run_w4a4(M, N, K, dtype)
    else:
        act_fp, act_deq, wgt_fp, wgt_deq = _run_w8a8(M, N, K, dtype)

    s_act = error_stats(act_fp, act_deq)
    s_wgt = error_stats(wgt_fp, wgt_deq)

    print_stats("activation", s_act)
    print_stats("weight", s_wgt)

    print("\n[activation sample] ref vs deq (row0, first 8)")
    print(torch.stack([act_fp[0, :8].float(), act_deq[0, :8].float()], dim=0))

    print("\n[weight sample] ref vs deq (row0, first 8)")
    print(torch.stack([wgt_fp[0, :8].float(), wgt_deq[0, :8].float()], dim=0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
