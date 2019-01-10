#!/usr/bin/env python

# REF [site] >>
#	http://docs.cython.org/en/latest/src/userguide/
#	https://cython.readthedocs.io/en/latest/src/tutorial/

# REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
def cython_tutorial():
	import helloworld

	import primes_c, primes_cpp
	
	print('Primes (C):', primes_c.primes(5))
	print('Primes (C++):', primes_cpp.primes(5))

# REF [site] >> http://docs.cython.org/en/latest/src/userguide/language_basics.html
def language_basic():
	pass

# REF [site] >> https://cython.readthedocs.io/en/latest/src/tutorial/external.html
# REF [site] >> https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html
def c_standard_lirary():
	import pyclibrary

	a = 1
	b = pyclibrary.f(a)
	print('f({}) = {}'.format(a, b))

	a = 0
	b = pyclibrary.cos(a)
	print('cos({}) = {}'.format(a, b))

def cpp_standard_lirary():
	import pycpplibrary

	a = 0
	b = pycpplibrary.cos(a)
	print('cos({}) = {}'.format(a, b))

# REF [site] >> https://cython.readthedocs.io/en/latest/src/tutorial/external.html
def cpp_function_example():
	import pyarithmetic

	a, b = 1, 2
	c = pyarithmetic.add(a, b)
	print('{} + {} = {}'.format(a, b, c))

	a, b = 5, 3
	c = pyarithmetic.sub(a, b)
	print('{} - {} = {}'.format(a, b, c))

	a, b = 2, 3
	c = pyarithmetic.mul(a, b)
	print('{} * {} = {}'.format(a, b, c))

	a, b = 7, 2
	c = pyarithmetic.div(a, b)
	print('{} / {} = {}'.format(a, b, c))

# REF [site] >> http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html
def cpp_class_example():
	import pyrectangle

	x0, y0, x1, y1 = 1, 2, 3, 4
	rect = pyrectangle.PyRectangle(x0, y0, x1, y1)

	print(dir(rect))
	print('x0 = {}, y0 = {}, x1 = {}, y1 = {}'.format(rect.x0, rect.y0, rect.x1, rect.y1))

# REF [site] >> http://docs.cython.org/en/latest/src/userguide/wrapping_CPlusPlus.html
def cpp_stl_example():
	import stl_cpp

	stl_cpp.vector_test()
	stl_cpp.string_test()

def main():
	cython_tutorial()

	language_basic()
	c_standard_lirary()
	cpp_standard_lirary()

	cpp_function_example()
	cpp_class_example()
	cpp_stl_example()

#%%------------------------------------------------------------------

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup.py build_ext --inplace
#	python cython_test.py

if '__main__' == __name__:
	main()
