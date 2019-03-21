#!/usr/bin/env python

import os, sys, platform
import traceback

#%%------------------------------------------------------------------
# Platform.
def platform_test():
	os.name

	sys.platform

	platform.platform()
	platform.system()
	platform.machine()

	platform.uname()
	platform.release()
	platform.version()

	platform.dist()
	platform.linux_distribution()
	platform.mac_ver()

#%%------------------------------------------------------------------
# Assert.
def assert_test():
	#assert(2 + 2 == 5, "Error: addition.")  # Error: not working.
	assert 2 + 2 == 5, "Error: addition."

	if __debug__:
		if not 2 + 2 == 5:
			raise AssertionError
			#raise AssertionError, "Error: addition."  # Error: invalid syntax.

#%%------------------------------------------------------------------
# Exception.
def exception_test():
	if not os.path.exists(prediction_dir_path):
		try:
			os.makedirs(prediction_dir_path)
		except OSError as ex:
			if ex.errno != os.errno.EEXIST:
				raise
		except:
			#ex = sys.exc_info()  # (type, exception object, traceback).
			##print('{} raised: {}.'.format(ex[0], ex[1]))
			#print('{} raised: {}.'.format(ex[0].__name__, ex[1]))
			#traceback.print_tb(ex[2], limit=None, file=sys.stdout)
			#traceback.print_exception(*sys.exc_info(), limit=None, file=sys.stdout)
			traceback.print_exc(limit=None, file=sys.stdout)

#%%------------------------------------------------------------------
# Lambda expression.
def lambda_expression():
	def make_incrementor(n):
		return lambda x: x + n

	increment_func = make_incrementor(5)
	print(increment_func(1))
	print(increment_func(3))

	a = [1, 2, 3]
	b = [4, 5, 6]
	c = [7, 8, 9]
	a_plus_b_plus_c = list(map(lambda x, y, z: x + y + z, a, b, c))
	print('a + b + c =', a_plus_b_plus_c)

	null_func = lambda x, y: None
	print('null_func is called:', null_func(2, 3))

	#func2 = lambda x, y: x, y  # NameError: name 'y' is not defined.
	func2 = lambda x, y: (x, y)
	print('func2 is called:', func2(2, 3))

#%%------------------------------------------------------------------
# Map, filter, reduce.
def map_filter_reduce():
	items = [1, 2, 3, 4, 5]
	squared = map(lambda x: x**2, items)  # class 'map'.
	print('Type of squared =', squared)
	print('squared =', list(squared))

	def mul(x):
		return x * x
	def add(x):
		return x + x

	funcs = [mul, add]
	for i in range(1, 6):
		value = list(map(lambda x: x(i), funcs))
		print(value)

	#--------------------
	number_list = range(-5, 5)
	less_than_zero = filter(lambda x: x < 0, number_list)  # class 'filter'.
	print('Type of less_than_zero =', less_than_zero)
	print('less_than_zero =', list(less_than_zero))

	#--------------------
	from functools import reduce

	items = [3, 4, 5, 6, 7]
	summation = reduce((lambda x, y: x + y), items)
	print('summation =', summation)
	product = reduce((lambda x, y: x * y), items)
	print('product =', product)

# REF [site] >>
#	https://docs.python.org/3/reference/datamodel.html#with-statement-context-managers
#	https://docs.quantifiedcode.com/python-anti-patterns/correctness/exit_must_accept_three_arguments.html
def with_statement_test():
	class Guard(object):
		def __init__(self, i):
			self._i = i
			print('Guard was constructed.')

		def __del__(self):
			print('Guard was destructed.')

		def __enter__(self):
			print('Guard was entered.')
			return self

		def __exit__(self, exception_type, exception_value, traceback):
			print('Guard was exited.')

		def func(self, d):
			print('Guard.func() was called: {}, {}'.format(self._i, d))

	print('Step #1.')
	with Guard(1) as guard:
		print('Step #2.')
		if guard is None:
			print('guard is None.')
		else:
			guard.func(2.0)
		print('Step #3.')
	print('Step #4.')

def func(i, f, s):
	print(i, f, s)

class func_obj(object):
	def __init__(self, ii):
		self._ii = ii

	def __call__(self, i, f, s):
		print(i + self._ii, f, s)

def caller_func(callee):
	callee(1, 2.0, 'abc')

def main():
	#platform_test()

	#assert_test()
	#exception_test()

	lambda_expression()
	#map_filter_reduce()

	#with_statement_test()

	#caller_func(func)
	#caller_func(func_obj(2))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
