from setuptools import setup, Extension
import os
import sys

tokenizer_module = Extension(
    'campusgpt_tokenizer',
    sources=['src/tokenizer/fast_tokenizer.c', 'src/tokenizer/tokenizer_bindings.c'],
    include_dirs=['src/tokenizer'],
    extra_compile_args=['-O3', '-march=native'] if sys.platform != 'win32' else ['/O2'],
)

extensions = [tokenizer_module]

# check if CUDA is available
try:
    import torch
    if torch.cuda.is_available():
        cuda_available = True
    else:
        cuda_available = False
except ImportError:
    cuda_available = False

if cuda_available and os.path.exists('src/quantization/nf4_kernels.cu'):
    print("CUDA available - building quantization kernels")
    # quantization will be built separately with CMake/nvcc
else:
    print("CUDA not available - skipping quantization kernels")

setup(
    name='campusgpt-native',
    version='0.1.0',
    description='Native C/CUDA components for CampusGPT',
    ext_modules=extensions,
    zip_safe=False,
)
