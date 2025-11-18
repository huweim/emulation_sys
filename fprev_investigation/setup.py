from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

# Get the current directory
this_dir = os.path.dirname(os.path.abspath(__file__))

# Collect source files
cpp_src = os.path.join(this_dir, "cuda", "binding.cpp")
cu_srcs = glob.glob(os.path.join(this_dir, "cuda", "*.cu"))

# Remove duplicates and maintain order
seen = set()
sources = [cpp_src] + [s for s in cu_srcs if not (s in seen or seen.add(s))]

if not cu_srcs:
    print("[WARN] No CUDA sources found. Check your .cu file locations!")

# Check if CUDA is available
try:
    import torch

    cuda_available = torch.cuda.is_available()

    # Get PyTorch library directory
    torch_lib_dir = os.path.join(torch.__path__[0], "lib")
    if not os.path.exists(torch_lib_dir):
        # Try alternative path
        torch_lib_dir = os.path.dirname(torch.__file__)
        if os.path.exists(os.path.join(torch_lib_dir, "lib")):
            torch_lib_dir = os.path.join(torch_lib_dir, "lib")

    print(f"PyTorch library directory: {torch_lib_dir}")

except ImportError:
    cuda_available = False
    torch_lib_dir = None

if not cuda_available:
    print("[WARN] CUDA not available. Building CPU-only version.")

# Configure CUDA extension
if cuda_available:
    torch_lib_abs = os.path.abspath(torch_lib_dir)

    # Find CUDA library path (important for cublas)
    # Try to get CUDA home from environment or PyTorch
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Try to infer from nvcc
        try:
            import subprocess

            nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        except:
            cuda_home = "/usr/local/cuda"  # fallback

    cuda_lib_dir = os.path.join(cuda_home, "lib64")
    if not os.path.exists(cuda_lib_dir):
        cuda_lib_dir = os.path.join(cuda_home, "lib")

    extension_kwargs = {
        "name": "fprev_cuda",
        "sources": sources,
        "include_dirs": [
            this_dir,
            os.path.join(this_dir, "include"),
        ],
        "library_dirs": [
            torch_lib_dir,
            cuda_lib_dir,
        ],
        "libraries": ["cublas"],
        "extra_compile_args": {
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-std=c++14",
                "--expt-relaxed-constexpr",
            ],
        },
        "extra_link_args": [
            f"-Wl,-rpath,{torch_lib_abs}",
            f"-Wl,-rpath,{cuda_lib_dir}",  # ensure runtime can find libcublas.so
        ],
    }
    ext_modules = [CUDAExtension(**extension_kwargs)]

setup(
    name="fprev_investigation",
    version="1.0.0",
    description="FMA Sequence Investigation using CUDA and FPRev Algorithm",
    author="FPRev Investigation Team",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if cuda_available else {},
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "networkx>=2.5",
        "pybind11>=2.6.0",
        "pydot",
        "graphviz",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "fprev-investigate=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
