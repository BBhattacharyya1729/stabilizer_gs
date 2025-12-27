from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "gf2_linalg",
        ["gf2_linalg.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "pauli_methods_c",
        ["pauli_methods_c.pyx"],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
    "triple_methods",
    ["triple_methods.pyx"],
    include_dirs=[numpy.get_include()],
),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )
)
