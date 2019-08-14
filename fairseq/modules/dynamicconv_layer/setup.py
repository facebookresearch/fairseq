from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='dynamicconv_layer',
    ext_modules=[
        CUDAExtension(
            name='dynamicconv_cuda',
            sources=[
                'dynamicconv_cuda.cpp',
                'dynamicconv_cuda_kernel.cu',
            ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
