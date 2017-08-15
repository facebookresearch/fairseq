from setuptools import setup, Extension
import sys
from torch.utils.ffi import create_extension


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/clib/libbleu.cpp',
        'fairseq/clib/module.cpp',
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
        version='0.1.0',
        description='Facebook AI Research Sequence-to-Sequence Toolkit',
        long_description=readme,
        license=license,
        install_requires=reqs.strip().split('\n'),
        packages=['fairseq'],
        ext_modules=[bleu],
    )
    conv_tbc.build()


if __name__ == '__main__':
    main()
