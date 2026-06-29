from setuptools import setup, Extension
import numpy as np

ext = Extension(
    'arimagarch._arimagarch',
    sources=[
        'src/arimagarch/_ext/arimagarch.c',
        'src/arimagarch/_ext/log_likelihood.c',
        'src/arimagarch/_ext/gradient.c',
    ],
    include_dirs=[
        np.get_include(),
        'src/arimagarch/_ext', 
    ],
    extra_compile_args=['-O3', '-march=native', '-ffast-math'],
)

setup(
    ext_modules=[ext],
)