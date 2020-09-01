from setuptools import setup, find_packages, Extension
from distutils.command.sdist import sdist as _sdist


try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True


if use_cython:
    extension = 'pyx'
else:
    extension = 'cpp'


extensions = [
    Extension(
        'fastBPE',
        [ "fastBPE/fastBPE." + extension ],
        language='c++',
        extra_compile_args=[
            "-std=c++11", "-Ofast", "-pthread"
        ],
    ),
]
if use_cython:
    extensions = cythonize(extensions)


with open('README.md') as f:
    readme = f.read()


setup(
    name = 'fastBPE',
    version = '0.1.0',
    description = 'C++ implementation of Neural Machine Translation of Rare Words with Subword Units, with Python API.',
    url = 'https://github.com/glample/fastBPE',
    long_description = readme,
    long_description_content_type = 'text/markdown',
    ext_package = '',
    ext_modules = extensions,
    packages=[
        'fastBPE',
    ],
)
