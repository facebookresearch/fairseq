import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmpt",
    version="0.0.1",
    author="Hu Xu, Po-yao Huang",
    author_email="huxu@fb.com",
    description="A package for multimodal pretraining.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pytorch/fairseq/examples/MMPT",
    packages=setuptools.find_packages(),
    install_requires=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: CC-BY-NC",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
