#!/usr/bin/env python

# REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
def cython_tutorial():
	import helloworld

	import primes_c, primes_cpp
	
	print('Primes (C):', primes_c.primes(5))
	print('Primes (C++):', primes_cpp.primes(5))

# REF [site] >> http://docs.cython.org/en/latest/src/userguide/language_basics.html
def language_basic():
	pass

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

	cpp_class_example()
	cpp_stl_example()

#%%------------------------------------------------------------------

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup.py build_ext --inplace
#	python cython_test.py

if '__main__' == __name__:
	main()

