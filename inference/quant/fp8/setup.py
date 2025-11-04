from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, glob

home = os.path.expanduser("~")
this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, ".."))

# 收集源文件：binding.cpp + 递归查找当前目录下所有 .cu
cpp_src = os.path.join(this_dir, "binding.cpp")
# cu_srcs = []
# cu_srcs += glob.glob(os.path.join(this_dir, "*.cu"))
cu_srcs = glob.glob(os.path.join(this_dir, "kernel", "*.cu"), recursive=True)
# 去重并保序
seen = set()
sources = [cpp_src] + [s for s in cu_srcs if not (s in seen or seen.add(s))]

if not cu_srcs:
    print("[WARN] No CUDA sources found. Check your .cu file locations!")

setup(
    name="fp8",
    version="0.1.0",
    # packages=[],  # 不打包 Python 包，仅安装扩展
    py_modules=["fp8_pseudo_quantize","mm"],
    ext_modules=[
        CUDAExtension(
            name="scaled_fp8_ops",
            sources=sources,
            include_dirs=[
                this_dir,  # 允许 #include "xxx.h" 就近查找
                os.path.join(this_dir, "include"),
                os.path.join(home, "cutlass", "include"),
                os.path.join(home, "cutlass", "tools/util/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-DENABLE_FP8",#new
                    "--extended-lambda",#new
                    "-gencode=arch=compute_120a,code=sm_120a",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=2.8.0"],
    python_requires=">=3.8",
)
