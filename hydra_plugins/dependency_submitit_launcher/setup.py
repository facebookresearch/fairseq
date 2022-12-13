# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from pathlib import Path

from read_version import read_version
from setuptools import find_namespace_packages, setup

setup(
    name="dependency-submitit-launcher",
    version=read_version("hydra_plugins/dependency_submitit_launcher", "__init__.py"),
    author="Alexei Baevski",
    author_email="abaevski@fb.com",
    description="Dependency-supporting Submitit Launcher for Hydra apps",
    packages=find_namespace_packages(include=["hydra_plugins.*"]),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Development Status :: 4 - Beta",
    ],
    install_requires=[
        "hydra-core>=1.0.4",
        "submitit>=1.0.0",
    ],
    include_package_data=True,
)
