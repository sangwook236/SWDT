from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# REF [site] >> https://docs.python.org/3/distutils/setupscript.html
# REF [site] >> https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html

"""
ext_modules = [
	Extension('cython_ext',
		sources=['helloworld.pyx', 'primes_c.pyx', 'primes_cpp.pyx',
			 'pyclibrary.pyx', 'pycpplibrary.pyx',
			 'pyarithmetic.pyx', 'pyrectangle.pyx', 'stl_cpp.pyx'],
		#libraries=['m']  # Unix-like specific.
	),
]

setup(
    name='cython_test',
	ext_modules = cythonize(ext_modules, annotate=True),
)
"""

setup(
    name='cython_test',
	ext_modules = cythonize(
		['helloworld.pyx', 'primes_c.pyx', 'primes_cpp.pyx',
		 'pyclibrary.pyx', 'pycpplibrary.pyx',
		 'pyarithmetic.pyx', 'pyrectangle.pyx', 'stl_cpp.pyx'],
		annotate=True),
)
