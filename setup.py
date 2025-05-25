#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension

# Check if we're on Windows
is_windows = sys.platform == "win32"

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


def write_version_py():
    with open(os.path.join("fairseq", "version.txt")) as f:
        version = f.read().strip()

    # write version info to fairseq/version.py
    with open(os.path.join("fairseq", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]

# Add Windows-specific configurations
cuda_lib_path = None
cuda_include_path = None

if "CUDA_HOME" in os.environ:
    cuda_home = os.environ.get("CUDA_HOME")
    if is_windows:
        # On Windows, some libraries are in bin instead of lib/x64
        cuda_lib_path = [os.path.join(cuda_home, "lib", "x64"), 
                         os.path.join(cuda_home, "bin")]
        cuda_include_path = [os.path.join(cuda_home, "include")]


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

extensions.extend(
    [
        cpp_extension.CppExtension(
            "fairseq.libbase",
            sources=[
                "fairseq/clib/libbase/balanced_assignment.cpp",
            ],
        ),
        cpp_extension.CppExtension(
            "fairseq.libnat",
            sources=[
                "fairseq/clib/libnat/edit_dist.cpp",
            ],
        ),
    ]
)

# Add CUDA extensions if CUDA is available
if "CUDA_HOME" in os.environ and not is_windows:
    # Configure CUDA extension settings based on platform
    cuda_extension_args = {}
    
    # Special handling for Windows
    if is_windows:
        cuda_home = os.environ.get("CUDA_HOME")
        # Manual specification of libraries and include directories
        cuda_extension_args = {
            'library_dirs': [
                os.path.join(cuda_home, 'lib', 'x64'),
                os.path.join(cuda_home, 'bin'),
            ],
            'libraries': ['cudart'],
            'define_macros': [('TORCH_EXTENSION_NAME', 'fairseq_cuda_extension')],
        }
    
    # Add CUDA extensions with platform-specific settings
    extensions.extend([
        cpp_extension.CppExtension(
            "fairseq.libnat_cuda",
            sources=[
                "fairseq/clib/libnat_cuda/edit_dist.cu",
                "fairseq/clib/libnat_cuda/binding.cpp",
            ],
            **cuda_extension_args
        ),
        cpp_extension.CppExtension(
            "fairseq.ngram_repeat_block_cuda",
            sources=[
                "fairseq/clib/cuda/ngram_repeat_block_cuda.cpp",
                "fairseq/clib/cuda/ngram_repeat_block_cuda_kernel.cu",
            ],
            **cuda_extension_args
        ),
    ])

# Customize build_ext class for Windows
if is_windows:
    from torch.utils.cpp_extension import BuildExtension

    class CustomBuildExtension(BuildExtension):
        def build_extensions(self):
            # Define specific compiler flags for Windows MSVC
            for extension in self.extensions:
                if hasattr(extension, 'sources') and any(source.endswith('.cu') for source in extension.sources):
                    self.compiler.compiler_so.append('/EHsc')
                    self.compiler.compiler_so.append('/MD')
            
            BuildExtension.build_extensions(self)
    
    cmdclass = {"build_ext": CustomBuildExtension}
else:
    cmdclass = {"build_ext": cpp_extension.BuildExtension}

if "READTHEDOCS" in os.environ:
    # don't build extensions when generating docs
    extensions = []
    if "build_ext" in cmdclass:
        del cmdclass["build_ext"]

    # use CPU build of PyTorch
    dependency_links = [        
        "https://download.pytorch.org/whl/cpu-cxx11-abi/torch-2.1.0%2Bcpu.cxx11.abi-cp310-cp310-linux_x86_64.whl"
    ]
else:
    dependency_links = []


if "clean" in sys.argv[1:]:
    # Source: https://github.com/pytorch/pytorch/blob/master/setup.py
    from setuptools.command.clean import clean

    class clean_with_subdirs(clean):
        def run(self):
            import glob
            import shutil

            with open(".gitignore", "r") as f:
                ignores = f.read()
                for wildcard in filter(bool, ignores.split("\n")):
                    for filename in glob.glob(wildcard):
                        try:
                            shutil.rmtree(filename)
                        except OSError:
                            try:
                                os.remove(filename)
                            except OSError:
                                pass

    cmdclass["clean"] = clean_with_subdirs


extra_packages = []
if os.path.exists(os.path.join("fairseq", "model_parallel", "megatron", "mpu")):
    extra_packages.append("fairseq.model_parallel.megatron.mpu")


def do_setup(package_data):
    setup(
        name="fairseq",
        version=version,
        description="Facebook AI Research Sequence-to-Sequence Toolkit",
        url="https://github.com/pytorch/fairseq",
        classifiers=[
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
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
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "torch>=2.0.0",
            "tqdm",
            "bitarray",
            "torchaudio>=0.8.0",
            "scikit-learn",
            "packaging",
        ],
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
        python_requires=">=3.8",
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


if __name__ == "__main__":
    package_data = {
        "fairseq": get_files(os.path.join("fairseq", "config"))
    }
    do_setup(package_data)
