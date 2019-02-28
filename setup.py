#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from setuptools import setup, find_packages, Extension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')

with open('README.md') as f:
    readme = f.read()


bleu = Extension(
    'fairseq.libbleu',
    sources=[
        'fairseq/clib/libbleu/libbleu.cpp',
        'fairseq/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)


setup(
    name='fairseq',
    version='0.6.1',
    description='Facebook AI Research Sequence-to-Sequence Toolkit',
    url='https://github.com/pytorch/fairseq',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=readme,
    install_requires=[
        'cffi',
        'numpy',
        'sacrebleu',
        # don't include torch, to support both release and nightly builds
        #'torch',
        'tqdm',
    ],
    packages=find_packages(exclude=['scripts', 'tests']),
    ext_modules=[bleu],
    test_suite='tests',
    entry_points={
        'console_scripts': [
            'fairseq-eval-lm = fairseq_cli.eval_lm:cli_main',
            'fairseq-generate = fairseq_cli.generate:cli_main',
            'fairseq-interactive = fairseq_cli.interactive:cli_main',
            'fairseq-preprocess = fairseq_cli.preprocess:cli_main',
            'fairseq-train = fairseq_cli.train:cli_main',
            'fairseq-score = fairseq_cli.score:main',
        ],
    },
)
