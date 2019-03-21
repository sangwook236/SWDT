#!/usr/bin/env python

# REF [site] >>
#	https://docs.python.org/3/library/timeit.html
#	https://www.geeksforgeeks.org/timeit-python-examples/

import timeit, math

def simple_statement():
	print('Using generator (?) =', timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))
	print('Using list =', timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000))
	print('Using map =', timeit.timeit('"-".join(map(str, range(100)))', number=10000))

def sqrt(x):
	return math.sqrt(x)

def repeat_statement():
	SETUP_CODE = """
from __main__ import sqrt
	"""
	TEST_CODE = """
map(sqrt, range(100))
	"""

	print('Repeat statement =', timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=5, number=10000))

def functor_speed():
	print('Lambda function =', timeit.timeit("""
lambda_func = lambda x: math.sqrt(x)
map(lambda_func, range(100))
	""", number=10000))
	print('General function =', timeit.timeit("""
def func(x):
	return math.sqrt(x)
map(func, range(100))
	""", number=10000))
	print('Function object =', timeit.timeit("""
class Func(object):
	def __call__(self, x):
		return math.sqrt(x)
map(Func(), range(100))
	""", number=10000))

def main():
	simple_statement()
	repeat_statement()

	functor_speed()

#%%------------------------------------------------------------------

# Usage:
#	python -m timeit '"-".join(str(n) for n in range(100))'
#	python -m timeit '"-".join([str(n) for n in range(100)])'
#	python -m timeit '"-".join(map(str, range(100)))'
#	python -m timeit -s "from math import sqrt" -n 10000 -r 5 "x = sqrt(1234567890)"

if '__main__' == __name__:
	main()
