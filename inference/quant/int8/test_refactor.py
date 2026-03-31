"""
INT8 test script mirroring inference.quant.int4.test_refactor.py structure.
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
    from inference.quant.int8 import ops, pseudo_quant  # type: ignore
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))
    from inference.quant.int8 import ops, pseudo_quant  # type: ignore


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
    def _safe_rmse(x: torch.Tensor, y: torch.Tensor) -> float:
        diff = x.float() - y.float()
        finite_mask = torch.isfinite(diff)
        if not finite_mask.any():
            return float("inf")
        diff_finite = diff[finite_mask]
        return torch.sqrt((diff_finite * diff_finite).mean()).item()

    @staticmethod
    def run_test_case(
        M: int,
        N: int,
        K: int,
        dist_a: str,
        dist_b: str,
        *,
        sample_count: int = 5,
        out_dtype: torch.dtype = torch.float16,
        clip_ratio: float = 1.0,
        model_kind: str = "emul",
    ) -> dict:
        a = DataGenerator.get_random_tensor((M, K), dist_a, dtype=out_dtype)
        b = DataGenerator.get_random_tensor((N, K), dist_b, dtype=out_dtype)

        a_q, a_s = ops.sym_int8_quant(a, scales=None, clip_ratio=clip_ratio)
        b_q, b_s = ops.sym_int8_quant(b, scales=None, clip_ratio=clip_ratio)
        real_output = ops.int8_linear(a_q, b_q, a_s, b_s, out_dtype=out_dtype)

        a_pseudo = pseudo_quant.int8_pseudo_quantize(a, scales=a_s, dequant_dtype=torch.float32)
        b_pseudo = pseudo_quant.int8_pseudo_quantize(b, scales=b_s, dequant_dtype=torch.float32)
        pseudo_res = (a_pseudo @ b_pseudo.t()).to(out_dtype)

        emul_res = pseudo_quant.int8_pseudo_emulation_gemm(
            a_q, b_q, a_s, b_s, out_dtype=out_dtype
        )

        diff_pseudo = (real_output.float() - pseudo_res.float()).abs()
        rmse_pseudo = TestRunner._safe_rmse(real_output, pseudo_res)
        diff_pseudo[real_output == pseudo_res] = 0.0
        both_nan = real_output.isnan() & pseudo_res.isnan()
        diff_pseudo[both_nan] = 0.0
        diff_pseudo[diff_pseudo.isnan()] = float("inf")
        max_diff_pseudo = diff_pseudo.max().item()

        diff_emul = (real_output.float() - emul_res.float()).abs()
        rmse_emul = TestRunner._safe_rmse(real_output, emul_res)
        diff_emul[real_output == emul_res] = 0.0
        both_nan_emul = real_output.isnan() & emul_res.isnan()
        diff_emul[both_nan_emul] = 0.0
        diff_emul[diff_emul.isnan()] = float("inf")
        max_diff_emul = diff_emul.max().item()

        if model_kind == "pseudo":
            active_diff = diff_pseudo
            max_diff = max_diff_pseudo
            model_res = pseudo_res
        else:
            active_diff = diff_emul
            max_diff = max_diff_emul
            model_res = emul_res

        status = "SUCCESS" if max_diff == 0 else "MISMATCH"
        mismatch_details: list[dict] = []
        if status == "MISMATCH":
            mismatch_indices = torch.nonzero(active_diff != 0, as_tuple=False)
            for idx in mismatch_indices[:sample_count]:
                r, c = idx[0].item(), idx[1].item()
                mismatch_details.append(
                    {
                        "idx": (r, c),
                        "real": real_output[r, c].item(),
                        "model": model_res[r, c].item(),
                        "diff": active_diff[r, c].item(),
                    }
                )

        return {
            "status": status,
            "max_diff": max_diff,
            "model_kind": model_kind,
            "rmse_pseudo": rmse_pseudo,
            "rmse_emul": rmse_emul,
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
        raise RuntimeError("CUDA is required to run this INT8 test.")

    seed = int(time.time() * 1e6) ^ int.from_bytes(os.urandom(8), "little")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed =", seed)

    num_iterations = int(os.environ.get("INT8_TEST_ITERS", "200"))
    distributions = ["normal", "uniform", "large", "outliers", "mixed_rows", "abs_large"]
    dims_m = [128, 256, 1024]
    dims_n = [128, 256, 1024]
    dims_k = [128, 256, 512, 1024]

    out_dtype_str = os.environ.get("INT8_TEST_OUT_DTYPE", "float16").lower().strip()
    out_dtype = torch.float16 if out_dtype_str in ("fp16", "float16") else torch.bfloat16
    clip_ratio = float(os.environ.get("INT8_TEST_CLIP_RATIO", "1.0"))
    model_kind = os.environ.get("INT8_TEST_MODEL", "emul").lower().strip()
    if model_kind not in ("pseudo", "emul"):
        raise ValueError(f"INT8_TEST_MODEL must be 'pseudo' or 'emul', got {model_kind!r}")

    sample_count = int(os.environ.get("INT8_TEST_PRINT_SAMPLES", "5"))
    max_mismatch_print = int(os.environ.get("INT8_TEST_MAX_MISMATCH_PRINT", "20"))
    mismatch_print_count = 0

    print(
        f"Starting INT8 real-vs-model verification ({num_iterations} iterations)...\n"
        f"  out_dtype={out_dtype}\n"
        f"  clip_ratio={clip_ratio}\n"
        f"  model_kind={model_kind}\n"
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
                M,
                N,
                K,
                da,
                db,
                sample_count=sample_count,
                out_dtype=out_dtype,
                clip_ratio=clip_ratio,
                model_kind=model_kind,
            )
            results.append(res)
            print(res["status"])
            if res["status"] == "MISMATCH" and mismatch_print_count < max_mismatch_print:
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
    print("VERIFICATION SUMMARY | INT8 real-vs-model")
    print(
        f"Total: {num_iterations}, Matches: {len(results)-len(mismatches)}, "
        f"Mismatches: {len(mismatches)}, model_kind={model_kind}"
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
