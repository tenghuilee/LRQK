from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os

extra_compile_args = {
    'cxx': [
        '-O3', 
        '-march=native', 
        '-mtune=native', 
        '-funroll-loops', 
        '-fopenmp', 
        '-mavx2', 
        '-lgomp', 
        '-DNDEBUG',
    ],
    'nvcc': ['-O3', '--use_fast_math']
}

setup(
    name='take_along_dim_grouped',
    ext_modules=[
        CUDAExtension(
            name='take_along_dim_grouped',
            sources=['take_along_dim_grouped.cpp'],
            extra_compile_args=extra_compile_args['cxx']
        ),
        # CUDAExtension(
        #     name="symmetric_matmul", 
        #     sources=["symmetric_matmul.cu"],
        #     extra_compile_args=extra_compile_args['nvcc'],
        # ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
