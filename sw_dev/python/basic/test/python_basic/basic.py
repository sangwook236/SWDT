#!/usr/bin/env python

import os, sys, platform
import traceback

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

x = 'global'

def variable_example():
	def func1():
		print('x =', x)

	def func2():
		x = x * 2  # UnboundLocalError: local variable 'x' referenced before assignment.
		print('x =', x)

	def func3():
		global x
		x = x * 2
		print('x =', x)

	def func4():
		x = 'local'
		x = x * 2
		print('x =', x)

	func1()
	#func2()
	func3()
	func4()

	#--------------------
	print('globals() =', globals())

	#--------------------
	# Functions can have attributes ((member) variables).

	def foo():
		foo.var = 10  # Not a local variable.
		var = 1  # A local variable.
		foo.num_calls += 1
		print(f'id(foo.var) = {id(foo.var)}, id(var) = {id(var)}')
		print(f'Call {foo.num_calls} of {foo.__name__!r}')
	foo.num_calls = 0  # Initialization. Similar to static variable in C/C++.

	#print('foo.var =', foo.var)  # AttributeError: 'function' object has no attribute 'var'.

	foo()
	foo()
	foo()

	print('foo.var =', foo.var)

def assert_test():
	#assert(2 + 2 == 5, "Error: addition.")  # Error: not working.
	assert 2 + 2 == 5, "Error: addition."

	if __debug__:
		if not 2 + 2 == 5:
			raise AssertionError
			#raise AssertionError, "Error: addition."  # Error: invalid syntax.

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

def lambda_expression():
	def make_incrementor(n):
		return lambda x: x + n

	print('Type of lambda expression =', type(make_incrementor))

	increment_func = make_incrementor(5)
	print(increment_func(1))
	print(increment_func(3))

	a = [1, 2, 3]
	b = [4, 5, 6]
	c = [7, 8, 9]
	a_plus_b_plus_c = list(map(lambda x, y, z: x + y + z, a, b, c))
	print('a + b + c =', a_plus_b_plus_c)

	noop_func = lambda x, y: None
	#def noop_func(x, y): pass
	print('noop_func is called:', noop_func(2, 3))

	#func2 = lambda x, y: x, y  # NameError: name 'y' is not defined.
	func2 = lambda x, y: (x, y)
	print('func2 is called:', func2(2, 3))

def map_filter_reduce():
	items = [1, 2, 3, 4, 5]
	squared = map(lambda x: x**2, items)  # class 'map'.
	print('Type of map() =', type(squared), squared)
	print('squared =', list(squared))

	def mul(x):
		return x * x
	def add(x):
		return x + x

	funcs = [mul, add]
	for i in range(1, 6):
		value = list(map(lambda x: x(i), funcs))
		print(value)

	map_with_two_args = map(lambda x, y: x + y, range(5), range(5))
	print('Mapping function with two argments =', list(map_with_two_args))

	#--------------------
	number_list = range(-5, 5)
	less_than_zero = filter(lambda x: x < 0, number_list)  # class 'filter'.
	print('Type of filter() =', type(less_than_zero), less_than_zero)
	print('less_than_zero =', list(less_than_zero))

	#--------------------
	from functools import reduce
	import operator

	items = [3, 4, 5, 6, 7]
	summation = reduce(lambda x, y: x + y, items)
	print('Type of reduce() =', type(summation), summation)  # The type of reduce() is its return type.
	print('Summation =', summation)
	product = reduce(lambda x, y: x * y, items)
	print('Product =', product)

	print('Max =', reduce(max, [5, 8, 3, 1]))
	print('Concatenation =', reduce(lambda s, x: s+str(x), [1, 2, 3, 4], ''))
	print('Flatten =', reduce(operator.concat, [[1, 2], [3, 4], [], [5]], []))

	difference = reduce(lambda x, y: x - y, items, 100)
	print('Difference =', difference)

	#--------------------
	# Chaining:
	#	Map -> filter -> reduce.
	#	Filter -> map -> reduce.

	print('Chaining =', reduce(lambda x, y: x + y, filter(lambda x: 0 == x % 2, map(lambda x: x**2, range(100)))))

	def evaluate_polynomial(a, x):
		xi = map(lambda i: x**i, range(0, len(a)))  # [x^0, x^1, x^2, ..., x^(n-1)].
		axi = map(operator.mul, a, xi)  # [a[0]*x^0, a[1]*x^1, ..., a[n-1]*x^(n-1)]
		return reduce(operator.add, axi, 0)
	print('Polynomial =', evaluate_polynomial([1, 2, 3, 4], 2))

# Context managers allow you to allocate and release resources precisely when you want to.
# The most widely used example of context managers is the with statement.
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

	variable_example()

	#assert_test()
	#exception_test()

	lambda_expression()
	map_filter_reduce()

	#with_statement_test()

	#caller_func(func)
	#caller_func(func_obj(2))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
