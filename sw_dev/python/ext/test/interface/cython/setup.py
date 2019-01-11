from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# REF [site] >>
#	https://cython.readthedocs.io/en/latest/index.html
#	https://cython.readthedocs.io/en/latest/src/userguide/index.html
#	https://cython.readthedocs.io/en/latest/src/tutorial/index.html
#	https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup.py build_ext --inplace

setup(
	name='cython_test',
	ext_modules=cythonize(
		['helloworld.pyx', 'primes_c.pyx', 'primes_cpp.pyx',
		 'pyclibrary.pyx', 'pycpplibrary.pyx',
		 'pyarithmetic.pyx', 'pyrectangle.pyx', 'stl_cpp.pyx'],
		annotate=False, quiet=True
	),
)
