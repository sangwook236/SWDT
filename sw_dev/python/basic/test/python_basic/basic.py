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

#%%------------------------------------------------------------------
# Map, filter, reduce.
def map_filter_reduce():
	items = [1, 2, 3, 4, 5]
	squared = map(lambda x: x**2, items)
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
	less_than_zero = filter(lambda x: x < 0, number_list)
	print('Type of less_than_zero =', less_than_zero)
	print('less_than_zero =', list(less_than_zero))

	#--------------------
	from functools import reduce

	items = [3, 4, 5, 6, 7]
	summation = reduce((lambda x, y: x + y), items)
	print('summation =', summation)
	product = reduce((lambda x, y: x * y), items)
	print('product =', product)

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

	#lambda_expression()
	#map_filter_reduce()

	caller_func(func)
	caller_func(func_obj(2))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
