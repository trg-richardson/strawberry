from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "strawberry_cython",
        sources=["strawberry_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]# Include NumPy's header files
    )
]

setup(
    name="strawberry_cython",
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
)
