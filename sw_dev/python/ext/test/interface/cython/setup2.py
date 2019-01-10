from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# REF [site] >>
#	https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

ext_modules=[
	Extension(
		'cython_lib.helloworld',
		sources=['helloworld.pyx'],
	),
	Extension(
		'cython_lib.primes_c',
		sources=['primes_c.pyx'],
	),
	Extension(
		'cython_lib.primes_cpp',
		sources=['primes_cpp.pyx'],
	),
	Extension(
		'cython_lib.pyclibrary',
		sources=['pyclibrary.pyx'],
	),
	Extension(
		'cython_lib.pycpplibrary',
		sources=['pycpplibrary.pyx'],
	),
	Extension(
		'cython_lib.pyarithmetic',
		sources=['pyarithmetic.pyx'],
	),
	Extension(
		'cython_lib.pyrectangle',
		sources=['pyrectangle.pyx'],
	),
	Extension(
		'cython_lib.stl_cpp',
		sources=['stl_cpp.pyx'],
	),
]

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup.py build_ext --inplace

setup(
	name='cython_lib',
	ext_modules=cythonize(ext_modules, annotate=False, quiet=True),
)
