#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from setuptools import Extension, find_packages, setup


if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


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
        "fairseq.libbleu",
        sources=[
            "fairseq/clib/libbleu/libbleu.cpp",
            "fairseq/clib/libbleu/module.cpp",
        ],
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    NumpyExtension(
        "fairseq.data.token_block_utils_fast",
        sources=["fairseq/data/token_block_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]


cmdclass = {}


try:
    # torch is not available when generating docs
    from torch.utils import cpp_extension

    extensions.extend(
        [
            cpp_extension.CppExtension(
                "fairseq.libnat",
                sources=[
                    "fairseq/clib/libnat/edit_dist.cpp",
                ],
            )
        ]
    )

    if "CUDA_HOME" in os.environ:
        extensions.extend(
            [
                cpp_extension.CppExtension(
                    "fairseq.libnat_cuda",
                    sources=[
                        "fairseq/clib/libnat_cuda/edit_dist.cu",
                        "fairseq/clib/libnat_cuda/binding.cpp",
                    ],
                )
            ]
        )
    cmdclass["build_ext"] = cpp_extension.BuildExtension

except ImportError:
    pass


if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]

    # use CPU build of PyTorch
    dependency_links = [
        "https://download.pytorch.org/whl/cpu/torch-1.3.0%2Bcpu-cp36-cp36m-linux_x86_64.whl"
    ]
else:
    dependency_links = []


if "clean" in sys.argv[1:]:
    # Source: https://bit.ly/2NLVsgE
    print("deleting Cython files...")
    import subprocess

    subprocess.run(
        ["rm -f fairseq/*.so fairseq/**/*.so fairseq/*.pyd fairseq/**/*.pyd"],
        shell=True,
    )


def do_setup(package_data):
    setup(
        name="fairseq",
        version="0.9.0",
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        setup_requires=[
            "cython",
            "numpy",
            "setuptools>=18.0",
        ],
        install_requires=[
            "cffi",
            "cython",
            "dataclasses",
            "editdistance",
            "hydra-core",
            "numpy",
            "regex",
            "sacrebleu>=1.4.12",
            "torch",
            "tqdm",
        ],
        dependency_links=dependency_links,
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
                "scripts",
                "scripts.*",
                "tests",
                "tests.*",
            ]
        ),
        package_data=package_data,
        ext_modules=extensions,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        cmdclass=cmdclass,
        zip_safe=False,
    )


def get_files(path, relative_to="fairseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


try:
    # symlink config and examples into fairseq package so package_data accepts them
    if "build_ext" not in sys.argv[1:]:
        os.symlink(os.path.join("..", "config"), "fairseq/config")
        os.symlink(os.path.join("..", "examples"), "fairseq/examples")
    package_data = {
        "fairseq": get_files("fairseq/config") + get_files("fairseq/examples"),
    }
    do_setup(package_data)
finally:
    if "build_ext" not in sys.argv[1:]:
        os.unlink("fairseq/config")
        os.unlink("fairseq/examples")
