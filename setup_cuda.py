from pathlib import Path
import os
import shutil

from setuptools import setup
from torch.utils import cpp_extension


MIN_TORCH_VERSION = (2, 3, 0)
MAX_TORCH_VERSION = (2, 4, 0)


def check_build_environment():
    """Check the CUDA toolchain before invoking the extension compiler."""
    import torch
    from torch.utils import cpp_extension as extension_utils

    version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:3])
    if version < MIN_TORCH_VERSION or version >= MAX_TORCH_VERSION:
        raise RuntimeError(
            "quant_cuda targets PyTorch 2.3.x; "
            f"found {torch.__version__}. Install requirements.txt first."
        )

    if torch.version.cuda is None:
        raise RuntimeError(
            "PyTorch was built without CUDA. Install a CUDA-enabled torch wheel first."
        )

    cuda_home = extension_utils.CUDA_HOME
    nvcc = shutil.which("nvcc") if cuda_home is None else str(Path(cuda_home) / "bin" / "nvcc")
    if nvcc is None or not Path(nvcc).exists():
        raise RuntimeError(
            "nvcc was not found. Install a CUDA 12.x toolkit or set CUDA_HOME. "
            "PyTorch 2.3.0 Linux wheels are built against CUDA 12.1."
        )

    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise RuntimeError(
            "No visible CUDA device was detected. Set TORCH_CUDA_ARCH_LIST, for example "
            "'export TORCH_CUDA_ARCH_LIST=8.0', to build without GPU detection."
        )

    print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"nvcc: {nvcc}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


check_build_environment()

setup(
    name="quant_cuda",
    version="2.3.0",
    description="CUDA kernels for 3-bit GPTQ inference",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "quant_cuda",
            ["quant_cuda.cpp", "quant_cuda_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
