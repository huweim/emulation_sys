from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _remove_unwanted_pytorch_nvcc_flags() -> None:
    # Keep half operators/conversions enabled for custom kernels.
    import torch.utils.cpp_extension as torch_cpp_ext

    remove = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in remove:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def _cuda_arch_flags() -> list[str]:
    env_arch = os.environ.get("INT4_CUDA_ARCH", "").strip()
    if env_arch:
        # Example: "80;86;90"
        arches = [a.strip() for a in env_arch.split(";") if a.strip()]
    else:
        arches = ["75", "80", "86"]
    flags: list[str] = []
    for a in arches:
        flags += ["-gencode", f"arch=compute_{a},code=sm_{a}"]
    return flags


@lru_cache(maxsize=1)
def load_int4_kernels():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build int4 kernels extension.")

    here = Path(__file__).resolve().parent
    int4_root = here.parent
    kernels_root = int4_root / "kernels"
    cutlass_root = int4_root / "third_party" / "cutlass"

    extra_include_paths = [
        str(kernels_root / "include"),
        str(cutlass_root / "include"),
        str(cutlass_root / "tools" / "util" / "include"),
    ]

    sources = [
        str(here / "bindings.cpp"),
        str(kernels_root / "src" / "gemm.cu"),
        str(kernels_root / "src" / "quant.cu"),
    ]

    _remove_unwanted_pytorch_nvcc_flags()

    extra_cuda_cflags = _cuda_arch_flags()

    return load(
        name="int4_kernels_cuda",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags + ["-O3"],
        extra_include_paths=extra_include_paths,
        with_cuda=True,
        verbose=bool(int(os.environ.get("INT4_BUILD_VERBOSE", "0"))),
    )

