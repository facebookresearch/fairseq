#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py
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
        'fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

conv_tbc = create_extension(
    'fairseq.temporal_convolution_tbc',
    relative_to='fairseq',
    headers=['fairseq/clib/temporal_convolution_tbc/temporal_convolution_tbc.h'],
    sources=['fairseq/clib/temporal_convolution_tbc/temporal_convolution_tbc.cpp'],
    define_macros=[('WITH_CUDA', None)],
    with_cuda=True,
    extra_compile_args=['-std=c++11'],
    source_extension='.cpp',
)


class build_py_hook(build_py):
    def run(self):
        conv_tbc.build()
        build_py.run(self)


setup(
    name='fairseq',
    version='0.2.0',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    long_description=readme,
    license=license,
    install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[bleu],

    # build and install PyTorch extensions
    package_data={
        'fairseq': ['temporal_convolution_tbc/*.so'],
    },
    include_package_data=True,
    cmdclass={
        'build_py': build_py_hook,
    },
)
