#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')


with open('README.md') as f:
    readme = f.read()


if sys.platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++', '-O3']
else:
    extra_compile_args = ['-std=c++11', '-O3']


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy
        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    Extension(
        'fairseq.libbleu',
        sources=[
            'fairseq/clib/libbleu/libbleu.cpp',
            'fairseq/clib/libbleu/module.cpp',
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'fairseq.data.data_utils_fast',
        sources=['fairseq/data/data_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        'fairseq.data.token_block_utils_fast',
        sources=['fairseq/data/token_block_utils_fast.pyx'],
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
    cpp_extension.CppExtension(
        'fairseq.libnat',
        sources=[
            'fairseq/clib/libnat/edit_dist.cpp',
        ],
    )
]


setup(
    name='fairseq',
    version='0.8.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    url='https://github.com/pytorch/fairseq',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'cython',
        'numpy',
        'setuptools>=18.0',
    ],
    install_requires=[
        'cffi',
        'cython',
        'fastBPE',
        'numpy',
        'regex',
        'sacrebleu',
        'torch',
        'tqdm',
    ],
    packages=find_packages(exclude=['scripts', 'tests']),
    ext_modules=extensions,
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'fairseq-eval-lm = fairseq_cli.eval_lm:cli_main',
            'fairseq-generate = fairseq_cli.generate:cli_main',
            'fairseq-interactive = fairseq_cli.interactive:cli_main',
            'fairseq-preprocess = fairseq_cli.preprocess:cli_main',
            'fairseq-score = fairseq_cli.score:main',
            'fairseq-train = fairseq_cli.train:cli_main',
            'fairseq-validate = fairseq_cli.validate:cli_main',
        ],
    },
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    zip_safe=False,
)
