from setuptools import setup, Extension


bleu = Extension(
    'libbleu',
    sources=[
        'clib/libbleu.cpp',
        'clib/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)


def main():
    setup(
        name='fairseq',
        ext_modules=[bleu],
    )


if __name__ == '__main__':
    main()
