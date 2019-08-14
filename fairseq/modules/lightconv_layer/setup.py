from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='lightconv_layer',
    ext_modules=[
        CUDAExtension('lightconv_cuda', [
            'lightconv_cuda.cpp',
            'lightconv_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
