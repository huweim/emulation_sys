from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

import os
import subprocess
import re


def _parse_nvcc_version() -> str | None:
    """
    Return nvcc version string like '12.4' if available.
    """
    try:
        cuda_home = os.environ.get("CUDA_HOME", None)
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
        out = subprocess.check_output([nvcc_path, "--version"], stderr=subprocess.STDOUT).decode("utf-8", errors="ignore")
        m = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", out)
        if not m:
            return None
        return m.group(1)
    except Exception:
        return None


def _sm_targets() -> list[str]:
    """
    Conservative default set for building these kernels.
    """
    nvcc_ver = _parse_nvcc_version()
    if nvcc_ver is None:
        # If nvcc version can't be detected, still try common archs.
        return ["75", "80", "86", "89", "120a", "121a"]

    ver = tuple(int(x) for x in nvcc_ver.split("."))
    support_sm120a = ver >= (12, 8)
    support_sm121a = ver >= (13, 0)

    # Follows nunchaku convention: 120a/121a are Blackwell.
    targets = ["75", "80", "86", "89"]
    if support_sm120a:
        targets.append("120a")
    if support_sm121a:
        targets.append("121a")
    return targets


current_dir = os.path.dirname(os.path.abspath(__file__))
nunchaku_dir = os.path.join(current_dir, "third_party", "nunchaku")


def _p(*parts: str) -> str:
    return os.path.join(*parts)


include_dirs = [
    _p(current_dir, "third_party", "spdlog_shim", "include"),
    _p(nunchaku_dir, "src"),
    _p(nunchaku_dir, "third_party", "cutlass", "include"),
]

sources = [
    os.path.join(current_dir, "binding.cpp"),
    # Torch <-> Tensor interop used by bindings.
    _p(nunchaku_dir, "src", "interop", "torch.cpp"),
    # W4A4 kernels (quantize + gemm + launch instantiations for int4).
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4.cu"),
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4_launch_fp16_int4.cu"),
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4_launch_fp16_int4_fasteri2f.cu"),
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4_launch_bf16_int4.cu"),
    # FP4 launch instantiations are still required at link-time because gemm_w4a4.cu
    # references quantize_w4a4_act_fuse_lora for USE_FP4=true.
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4_launch_fp16_fp4.cu"),
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w4a4_launch_bf16_fp4.cu"),
    # W8A8 (int8) kernels.
    _p(nunchaku_dir, "src", "kernels", "zgemm", "gemm_w8a8.cu"),
]


sm_targets = _sm_targets()
abi_flag = int(torch._C._GLIBCXX_USE_CXX11_ABI)
nvcc_flags = [
    "-O3",
    "-std=c++20",
    "-DFMT_HEADER_ONLY",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # Re-enable half/bfloat16 conversions/operators disabled by PyTorch defaults.
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}",
    "-Xcudafe",
    "--diag_suppress=20208",  # spdlog: long double warnings in device code
]
for t in sm_targets:
    nvcc_flags += ["-gencode", f"arch=compute_{t},code=sm_{t}"]


setup(
    name="w4a4_w8a8_int",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="w4a4_w8a8_int_ops",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20", "-DFMT_HEADER_ONLY", f"-D_GLIBCXX_USE_CXX11_ABI={abi_flag}"],
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.7.0"],
)

