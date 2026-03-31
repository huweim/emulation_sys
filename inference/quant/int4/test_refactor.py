"""
INT4 test script mirroring nvfp_kernel/test_refactor.py structure.

Goal: compare real INT4 kernel path vs pseudo-quant reference under identical
per-row scales, and check whether outputs are bit-exact in fp16/bf16.
"""

from __future__ import annotations

import os
import random
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    # Preferred: run from repo root, import as a package.
    from inference.quant.int4 import ops, pseudo_quant  # type: ignore
except ModuleNotFoundError:
    # Fallback: allow `python test_refactor.py` from this directory.
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from inference.quant.int4 import ops, pseudo_quant  # type: ignore


class DataGenerator:
    @staticmethod
    def get_random_tensor(
        shape: tuple[int, ...],
        dist_type: str,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        if dist_type == "normal":
            return torch.randn(shape, device=device, dtype=dtype)
        if dist_type == "uniform":
            return (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
        if dist_type == "large":
            return torch.randn(shape, device=device, dtype=dtype) * 100.0
        if dist_type == "small":
            return torch.randn(shape, device=device, dtype=dtype) * 0.001
        if dist_type == "outliers":
            t = torch.randn(shape, device=device, dtype=dtype)
            mask = torch.rand(shape, device=device) < 0.01
            t[mask] *= 50.0
            return t
        if dist_type == "mixed_rows":
            t = torch.randn(shape, device=device, dtype=dtype)
            scale = torch.exp(torch.randn(shape[0], 1, device=device) * 2)
            return t * scale.to(dtype)
        if dist_type == "abs_large":
            return torch.abs(torch.randn(shape, device=device, dtype=dtype)) * 100.0
        raise ValueError(f"Unknown distribution type: {dist_type}")


class TestRunner:
    @staticmethod
    def run_test_case(
        iter_idx: int,
        M: int,
        N: int,
        K: int,
        dist_a: str,
        dist_b: str,
        *,
        sample_count: int = 5,
        out_dtype: torch.dtype = torch.float16,
        clip_ratio: float = 1.0,
    ) -> dict:
        del iter_idx

        a = DataGenerator.get_random_tensor((M, K), dist_a, dtype=out_dtype)
        b = DataGenerator.get_random_tensor((N, K), dist_b, dtype=out_dtype)

        # -------- Real INT4 path --------
        a_q, a_s = ops.sym_int4_quant(a, scales=None, clip_ratio=clip_ratio)
        b_q, b_s = ops.sym_int4_quant(b, scales=None, clip_ratio=clip_ratio)
        real_output = ops.int4_packed_linear(a_q, b_q, a_s, b_s, out_dtype=out_dtype)

        # -------- Pseudo INT4 path --------
        # Use scales from the real quant path so the comparison is input-aligned.
        a_pseudo = pseudo_quant.int4_pseudo_quantize(a, scales=a_s, dequant_dtype=torch.float32)
        b_pseudo = pseudo_quant.int4_pseudo_quantize(b, scales=b_s, dequant_dtype=torch.float32)
        modeling_res = (a_pseudo @ b_pseudo.t()).to(out_dtype)

        # -------- Comparison (aligned with nvfp_kernel/test_refactor.py) --------
        diff = (real_output.float() - modeling_res.float()).abs()
        matches = real_output == modeling_res
        diff[matches] = 0.0
        both_nan = real_output.isnan() & modeling_res.isnan()
        diff[both_nan] = 0.0
        diff[diff.isnan()] = float("inf")

        max_diff = diff.max().item()
        status = "SUCCESS" if max_diff == 0 else "MISMATCH"
        mismatch_details: list[dict] = []

        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(diff != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                mismatch_details.append(
                    {
                        "idx": (r, c),
                        "real": real_output[r, c].item(),
                        "model": modeling_res[r, c].item(),
                        "diff": diff[r, c].item(),
                    }
                )

        return {
            "status": status,
            "max_diff": max_diff,
            "M": M,
            "N": N,
            "K": K,
            "dist_a": dist_a,
            "dist_b": dist_b,
            "out_dtype": str(out_dtype).replace("torch.", ""),
            "clip_ratio": clip_ratio,
            "mismatch_details": mismatch_details,
        }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this INT4 test.")

    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)

    num_iterations = int(os.environ.get("INT4_TEST_ITERS", "200"))
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    dims_m = [128, 256, 1024]
    dims_n = [128, 256, 1024]
    dims_k = [128, 256, 512, 1024]

    out_dtype_str = os.environ.get("INT4_TEST_OUT_DTYPE", "float16").lower().strip()
    out_dtype = torch.float16 if out_dtype_str in ("fp16", "float16") else torch.bfloat16

    clip_ratio = float(os.environ.get("INT4_TEST_CLIP_RATIO", "1.0"))

    NUM_SAMPLES_TO_PRINT = int(os.environ.get("INT4_TEST_PRINT_SAMPLES", "5"))
    MAX_MISMATCH_TESTS = int(os.environ.get("INT4_TEST_MAX_MISMATCH_PRINT", "20"))
    mismatch_print_count = 0

    print(
        f"Starting INT4 real-vs-pseudo verification ({num_iterations} iterations)...\n"
        f"  out_dtype={out_dtype}\n"
        f"  clip_ratio={clip_ratio}\n"
        f"  distributions={distributions}\n"
        f"  dims_m={dims_m}\n"
        f"  dims_n={dims_n}\n"
        f"  dims_k={dims_k}\n"
    )

    results: list[dict] = []
    for i in range(num_iterations):
        M, N, K = random.choice(dims_m), random.choice(dims_n), random.choice(dims_k)
        da, db = random.choice(distributions), random.choice(distributions)

        print(f"\rTest {i+1}/{num_iterations}: K={K:4}, A={da:10}, B={db:10} ... ", end="")
        sys.stdout.flush()

        try:
            res = TestRunner.run_test_case(
                i,
                M,
                N,
                K,
                da,
                db,
                sample_count=NUM_SAMPLES_TO_PRINT,
                out_dtype=out_dtype,
                clip_ratio=clip_ratio,
            )
            results.append(res)
            print(res["status"])

            if res["status"] == "MISMATCH" and mismatch_print_count < MAX_MISMATCH_TESTS:
                mismatch_print_count += 1
                print("    >>> Mismatch Details:")
                for d in res["mismatch_details"]:
                    print(
                        f"      Pos {d['idx']}: Real={d['real']:.6f} | "
                        f"Model={d['model']:.6f} | Diff={d['diff']:.6f}"
                    )
        except Exception as e:
            print(f"\nFATAL ERROR on Test {i+1}: {e}")
            traceback.print_exc()

    mismatches = [r for r in results if r["status"] == "MISMATCH"]
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY | INT4 real-vs-pseudo")
    print(
        f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, "
        f"Mismatches: {len(mismatches)}"
    )
    print("=" * 70)

    if len(mismatches) == 0:
        print("*** PERFECT MATCH (bit-exact) ***")
        return 0

    acc = (len(results) - len(mismatches)) / max(1, len(results)) * 100.0
    print(f"Bit-exact rate: {acc:.2f}%")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
