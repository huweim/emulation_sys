"""
W4A4 / W8A8 emulation alignment test runner.
Style aligned to nvfp_kernel/verify_acc_modeling.py but adapted to int kernels.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
import traceback

import torch
import w4a4_w8a8_int as ops

from emulation.core import IntKernelUtils
from emulation.kernel import emulated_gemm_w4a4, emulated_gemm_w8a8


def compare_outputs(real_output: torch.Tensor, emu_output: torch.Tensor):
    diff = (real_output.float() - emu_output.float()).abs()
    matches = real_output == emu_output
    diff[matches] = 0.0
    both_nan = real_output.isnan() & emu_output.isnan()
    diff[both_nan] = 0.0
    diff[diff.isnan()] = float("inf")
    return diff


def pseudo_w4a4_dequant_gemm(
    act_q: torch.Tensor,
    w_q: torch.Tensor,
    act_scales: torch.Tensor,
    w_scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference pseudo path: dequantize q tensors then do fp GEMM."""
    M, K2 = act_q.shape
    N, K2w = w_q.shape
    if K2 != K2w:
        raise ValueError("act_q and w_q shape mismatch on K/2")
    K = K2 * 2
    G = K // 64

    a_int4 = IntKernelUtils.unpack_int4_pairs_act(act_q, K).to(torch.float32)
    w_int4 = IntKernelUtils.unpack_int4_pairs_wgt(w_q, K).to(torch.float32)

    a_deq = (a_int4.view(M, G, 64) * act_scales.t().float().unsqueeze(-1)).reshape(M, K)
    w_deq = (w_int4.view(N, G, 64) * w_scales.t().float().unsqueeze(-1)).reshape(N, K)

    out = a_deq @ w_deq.t()
    if bias is not None:
        out = out + bias.float().view(1, N)
    return out.to(act_scales.dtype)


class TestRunner:
    @staticmethod
    def run_test_case_w4a4(M: int, N: int, K: int, dtype: torch.dtype, sample_count: int = 5, debug_mismatch: bool = False) -> dict:
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        w = torch.randn(N, K, device="cuda", dtype=dtype)
        bias = torch.randn(N, device="cuda", dtype=dtype)

        act_q, act_scales = ops.quantize_w4a4_act(x)
        w_q, w_scales = ops.quantize_w4a4_wgt(w)

        real_output = ops.gemm_w4a4(act_q, w_q, act_scales, w_scales, bias=bias)
        pseudo_output = pseudo_w4a4_dequant_gemm(act_q, w_q, act_scales, w_scales, bias=bias)

        if debug_mismatch:
            emu_output, debug_info = emulated_gemm_w4a4(act_q, w_q, act_scales, w_scales, bias=bias, return_debug=True)
        else:
            emu_output = emulated_gemm_w4a4(act_q, w_q, act_scales, w_scales, bias=bias)
            debug_info = None

        diff_emu = compare_outputs(real_output, emu_output)
        diff_pseudo = compare_outputs(real_output, pseudo_output)
        max_diff_emu = diff_emu.max().item()
        max_diff_pseudo = diff_pseudo.max().item()

        status = "SUCCESS" if max_diff_emu == 0 else "MISMATCH"

        mismatch_details = []
        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(diff_emu != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                item = {
                    "idx": (r, c),
                    "real": real_output[r, c].item(),
                    "model": emu_output[r, c].item(),
                    "pseudo": pseudo_output[r, c].item(),
                    "diff_model": diff_emu[r, c].item(),
                    "diff_pseudo": diff_pseudo[r, c].item(),
                }
                if debug_info is not None:
                    item["debug"] = {
                        "shape": debug_info.get("shape"),
                        "a_q_sample": debug_info.get("a_q_sample"),
                        "w_q_sample": debug_info.get("w_q_sample"),
                        "ascales_sample": debug_info.get("ascales_sample"),
                        "wscales_sample": debug_info.get("wscales_sample"),
                    }
                mismatch_details.append(item)

        return {
            "status": status,
            "max_diff": max_diff_emu,
            "max_diff_pseudo": max_diff_pseudo,
            "kind": "w4a4",
            "dtype": str(dtype),
            "M": M,
            "N": N,
            "K": K,
            "mismatch_details": mismatch_details,
        }

    @staticmethod
    def run_test_case_w8a8(M: int, N: int, K: int, sample_count: int = 5, debug_mismatch: bool = False) -> dict:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)

        act_q, act_scales = ops.quantize_w8a8_act(x, fuse_glu=False)
        w_q, w_scales, packed_bias = ops.quantize_w8a8_wgt(w, bias)

        real_output = ops.gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=packed_bias)

        if debug_mismatch:
            emu_output, debug_info = emulated_gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=packed_bias, packed_wgt=True, return_debug=True)
        else:
            emu_output = emulated_gemm_w8a8(act_q, w_q, act_scales, w_scales, bias=packed_bias, packed_wgt=True)
            debug_info = None

        diff = compare_outputs(real_output, emu_output)
        max_diff = diff.max().item()
        status = "SUCCESS" if max_diff == 0 else "MISMATCH"

        mismatch_details = []
        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                item = {"idx": (r, c), "real": real_output[r, c].item(), "model": emu_output[r, c].item(), "diff": diff[r, c].item()}
                if debug_info is not None:
                    item["debug"] = {
                        "shape": debug_info.get("shape"),
                        "packed_wgt": debug_info.get("packed_wgt"),
                        "act_q_sample": debug_info.get("act_q_sample"),
                        "w_q_sample": debug_info.get("w_q_sample"),
                        "ascales_sample": debug_info.get("ascales_sample"),
                        "wscales_sample": debug_info.get("wscales_sample"),
                    }
                mismatch_details.append(item)

        return {"status": status, "max_diff": max_diff, "kind": "w8a8", "dtype": "torch.bfloat16", "M": M, "N": N, "K": K, "mismatch_details": mismatch_details}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W4A4/W8A8 emulation alignment test")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--mode", choices=["all", "w4a4", "w8a8"], default="w4a4")
    p.add_argument("--print-samples", type=int, default=5)
    p.add_argument("--max-mismatch-print", type=int, default=20)
    p.add_argument("--debug-on-mismatch", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)

    dims_m = [256, 512, 1024]
    dims_n = [128, 256, 512, 1024]
    dims_k_w4a4 = [64, 128, 256, 512]
    dims_k_w8a8 = [64, 128, 256, 512]
    w4a4_dtypes = [torch.float16, torch.bfloat16]

    print("=" * 78)
    print("Starting W4A4/W8A8 emulation verification")
    print(f"iters={args.iters}, mode={args.mode}, debug_on_mismatch={args.debug_on_mismatch}")
    print("=" * 78)

    results = []
    mismatch_print_count = 0

    for i in range(args.iters):
        M = random.choice(dims_m)
        N = random.choice(dims_n)
        run_kind = ("w4a4" if i % 2 == 0 else "w8a8") if args.mode == "all" else args.mode

        if run_kind == "w4a4":
            K = random.choice(dims_k_w4a4)
            dtype = random.choice(w4a4_dtypes)
            desc = f"W4A4 dtype={dtype}"
        else:
            K = random.choice(dims_k_w8a8)
            dtype = torch.bfloat16
            desc = "W8A8"

        print(f"\rTest {i+1}/{args.iters}: {desc:24s} shape=({M},{N},{K}) ... ", end="")
        sys.stdout.flush()

        try:
            if run_kind == "w4a4":
                res = TestRunner.run_test_case_w4a4(M, N, K, dtype=dtype, sample_count=args.print_samples, debug_mismatch=args.debug_on_mismatch)
            else:
                res = TestRunner.run_test_case_w8a8(M, N, K, sample_count=args.print_samples, debug_mismatch=args.debug_on_mismatch)

            results.append(res)
            print(res["status"])

            if run_kind == "w4a4":
                print(f"    [W4A4 Ref] max_diff_pseudo={res['max_diff_pseudo']}")

            if res["status"] == "MISMATCH" and mismatch_print_count < args.max_mismatch_print:
                mismatch_print_count += 1
                print("    >>> Mismatch details:")
                for d in res["mismatch_details"]:
                    print(
                        f"      Pos {d['idx']}: Real={d['real']:.6f} | Model={d['model']:.6f} | "
                        f"Pseudo={d['pseudo']:.6f} | DiffModel={d['diff_model']:.6f} | DiffPseudo={d['diff_pseudo']:.6f}"
                    )
                    if "debug" in d and d["debug"] is not None:
                        print(f"        [Debug] {d['debug']}")

        except Exception as e:
            print(f"\nFATAL ERROR on test {i+1}: {e}")
            traceback.print_exc()
            results.append({"status": "EXCEPTION", "kind": run_kind, "M": M, "N": N, "K": K, "err": str(e)})

    mismatches = [r for r in results if r.get("status") == "MISMATCH"]
    exceptions = [r for r in results if r.get("status") == "EXCEPTION"]
    total = len(results)
    success = total - len(mismatches) - len(exceptions)

    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print(f"Total: {total}, Success: {success}, Mismatch: {len(mismatches)}, Exception: {len(exceptions)}")
    if total > 0:
        print(f"Accuracy (exact match): {success / total * 100:.2f}%")
    print("=" * 78)

    return 0 if len(mismatches) == 0 and len(exceptions) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
