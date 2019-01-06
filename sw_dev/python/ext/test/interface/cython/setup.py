from distutils.core import setup
from Cython.Build import cythonize

setup(
	ext_modules = cythonize(
		['helloworld.pyx', 'primes_c.pyx', 'primes_cpp.pyx', 'pyrectangle.pyx', 'stl_cpp.pyx'],
		annotate=True),
)
