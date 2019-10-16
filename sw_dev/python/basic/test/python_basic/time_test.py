#!/usr/bin/env python

# REF [site] >>
#	https://docs.python.org/3/library/timeit.html
#	https://www.geeksforgeeks.org/timeit-python-examples/

import time, timeit, math

def time_example():
	def foo(num_times):
		for _ in range(num_times):
			sum([i**2 for i in range(10000)])

	start_time = time.perf_counter()
	foo(100)
	end_time = time.perf_counter()
	run_time = end_time - start_time
	print(f'Finished {foo.__name__!r} in {run_time:.4f} secs.')

def timeit_simple_statement():
	print('Using generator (?) =', timeit.timeit('"-".join(str(n) for n in range(100))', number=10000))
	print('Using list =', timeit.timeit('"-".join([str(n) for n in range(100)])', number=10000))
	print('Using map =', timeit.timeit('"-".join(map(str, range(100)))', number=10000))

def udf(x):
	return math.sqrt(x)

def timeit_repeat_statement():
	setup_statement = """
from __main__ import udf
	"""
	statement = """
map(udf, range(100))
	"""
	print('Repeat statement =', timeit.repeat(stmt=statement, setup=setup_statement, repeat=5, number=10000))

def timeit_functor_speed():
	setup_statement = """
lambda_func = lambda x: math.sqrt(x)
	"""
	statement = """
map(lambda_func, range(100))
	"""
	print('Lambda function =', timeit.timeit(stmt=statement, setup=setup_statement, number=10000))

	setup_statement = """
def func(x):
	return math.sqrt(x)
	"""
	statement = """
map(func, range(100))
	"""
	print('General function =', timeit.timeit(stmt=statement, setup=setup_statement, number=10000))

	setup_statement = """
class Func(object):
	def __call__(self, x):
		return math.sqrt(x)
	"""
	statement = """
map(Func(), range(100))
	"""
	print('Function object =', timeit.timeit(stmt=statement, setup=setup_statement, number=10000))

def main():
	time_example()

	timeit_simple_statement()
	timeit_repeat_statement()

	timeit_functor_speed()

#--------------------------------------------------------------------

# Usage:
#	python -m timeit '"-".join(str(n) for n in range(100))'
#	python -m timeit '"-".join([str(n) for n in range(100)])'
#	python -m timeit '"-".join(map(str, range(100)))'
#	python -m timeit -s "from math import sqrt" -n 10000 -r 5 "x = sqrt(1234567890)"

if '__main__' == __name__:
	main()
