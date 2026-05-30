from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

# Get the directory of this setup.py file
ext_dir = os.path.dirname(os.path.abspath(__file__))
csrc_dir = os.path.join(ext_dir, 'csrc')
rust_lib_path = os.path.join(ext_dir, '../../../engines/rust/train_resizer/target/release/libtrain_resizer.a')

extra_objects = []
if os.path.exists(rust_lib_path):
    extra_objects.append(rust_lib_path)
    print(f"➜ Found Rust library: {rust_lib_path}")

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
                os.path.join(csrc_dir, 'fused_gcn.cu'),
                os.path.join(csrc_dir, 'fused_aggregation.cu'),
                os.path.join(csrc_dir, 'fused_bayesian_veg.cu'),
                os.path.join(csrc_dir, 'retinex.cu'),
                os.path.join(csrc_dir, 'inference_engine.cpp'),
            ],
            libraries=['cufft'],
            extra_objects=extra_objects,
            extra_compile_args={
                'cxx': ['-O3', '-mavx512f', '-mavx512dq'], # AVX-512 for fast taxonomy filtering
                'nvcc': ['-O3', '--use_fast_math', '-arch=sm_89'] 
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)