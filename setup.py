from setuptools import setup, Extension
from torch.utils.ffi import create_extension


bleu = Extension(
    'libbleu',
    sources=[
        'clib/libbleu.cpp',
        'clib/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

conv_tbc = create_extension(
    'clib.temporal_convolution_tbc',
    headers=['clib/temporal_convolution_tbc.h'],
    sources=['clib/temporal_convolution_tbc.cpp'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_compile_args=['-std=c++11'],
)


def main():
    setup(
        name='fairseq',
        ext_modules=[bleu],
    )
    conv_tbc.build()


if __name__ == '__main__':
    main()
