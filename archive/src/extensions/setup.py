from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

# Get the directory of this setup.py file
ext_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(ext_dir, 'csrc')

setup(
    name='plantclef_ext',
    ext_modules=[
        CUDAExtension(
            name='plantclef_ext',
            sources=[
                os.path.join(csrc_dir, 'bindings.cpp'),
                os.path.join(csrc_dir, 'fused_loss.cu'),
                os.path.join(csrc_dir, 'sahi_tiling.cu'),
                os.path.join(csrc_dir, 'fused_projection.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3', '-mavx512f', '-mavx512dq'], # AVX-512 for fast taxonomy filtering
                'nvcc': ['-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)