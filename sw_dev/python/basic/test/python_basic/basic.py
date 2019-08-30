#!/usr/bin/env python

import os, sys, platform, abc
import itertools, functools, operator
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

# REF [site] >> https://docs.python.org/3/library/itertools.html
def itertools_test():
	print('itertools.count(10) =', type(itertools.count(10)))
	print('itertools.count(10) =', end=' ')
	for item in itertools.count(start=10, step=2):
		print(item, end=', ')
		if item >= 20:
			break
	print()

	#--------------------
	print("itertools.cycle('ABCD') =", type(itertools.cycle('ABCD')))
	print("itertools.cycle('ABCD') =", end=' ')
	for idx, item in enumerate(itertools.cycle('ABCD')):
		print(item, end=', ')
		if idx >= 10:
			break
	print()

	#--------------------
	print('itertools.repeat(37, 7) =', type(itertools.repeat(37, 7)))
	print('itertools.repeat(37, 7) =', end=' ')
	for item in itertools.repeat(37, 7):
		print(item, end=', ')
	print()

	#--------------------
	print('itertools.accumulate([1, 2, 3, 4, 5]) =', type(itertools.accumulate([1, 2, 3, 4, 5])))
	print('itertools.accumulate([1, 2, 3, 4, 5]) =', end=' ')
	for item in itertools.accumulate([1, 2, 3, 4, 5]):
		print(item, end=', ')
	print()

	#--------------------
	print("itertools.groupby('AAAABBBCCDAABBB') =", type(itertools.groupby('AAAABBBCCDAABBB')))
	print("itertools.groupby('AAAABBBCCDAABBB'): keys =", list(k for k, g in itertools.groupby('AAAABBBCCDAABBB')))
	print("itertools.groupby('AAAABBBCCDAABBB'): groups =", list(list(g) for k, g in itertools.groupby('AAAABBBCCDAABBB')))

	#--------------------
	print("itertools.chain('ABC', 'DEF', 'ghi') =", list(itertools.chain('ABC', 'DEF', 'ghi')))
	print("itertools.chain.from_iterable(['ABC', 'DEF', 'ghi']) =", list(itertools.chain.from_iterable(['ABC', 'DEF', 'ghi'])))

	print("itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1]) =", list(itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1])))
	print("itertools.islice('ABCDEFG', 2, None) =", list(itertools.islice('ABCDEFG', 2, None)))

	print('itertools.starmap(pow, [(2, 5), (3, 2), (10, 3)]) =', list(itertools.starmap(pow, [(2, 5), (3, 2), (10, 3)])))

	#--------------------
	print('itertools.filterfalse(lambda x: x % 2, range(10)) =', list(itertools.filterfalse(lambda x: x % 2, range(10))))
	print('itertools.dropwhile(lambda x: x < 5, [1, 4, 6, 4, 1] =', list(itertools.dropwhile(lambda x: x < 5, [1, 4, 6, 4, 1])))
	print('itertools.takewhile(lambda x: x < 5, [1, 4, 6, 4, 1] =', list(itertools.takewhile(lambda x: x < 5, [1, 4, 6, 4, 1])))

	#--------------------
	print("itertools.zip_longest('ABCD', 'xy', fillvalue='-') =", list(itertools.zip_longest('ABCD', 'xy', fillvalue='-')))

	#--------------------
	print("itertools.tee('ABCDEFG', 2) =", itertools.tee('ABCDEFG', 2))

	#--------------------
	print("itertools.product('ABCD', repeat=2) =", list(itertools.product('ABCD', repeat=2)))
	print("itertools.permutations('ABCD', 2) =", list(itertools.permutations('ABCD', 2)))
	print("itertools.combinations('ABCD', 2) =", list(itertools.combinations('ABCD', 2)))
	print("itertools.combinations_with_replacement('ABCD', 2) =", list(itertools.combinations_with_replacement('ABCD', 2)))

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

	# More than two args.
	map_with_two_args = map(lambda x, y: x + y, range(5), range(5))
	print('Mapping a function with two argments =', list(map_with_two_args))

	map_with_three_args = map(lambda x, y, z: x + y + z, range(10), range(5), range(8))
	print('Mapping a function with three argments =', list(map_with_three_args))

	#--------------------
	number_list = range(-5, 5)
	less_than_zero = filter(lambda x: x < 0, number_list)  # class 'filter'.
	print('Type of filter() =', type(less_than_zero), less_than_zero)
	print('less_than_zero =', list(less_than_zero))

	# More than two args.
	#print('Max =', list(filter(lambda x, y: x + y <= 5, [1, 4, 5], [3, 2, 5])))  # TypeError: filter expected 2 arguments, got 3.
	print('Max =', list(filter(lambda xy: xy[0] + xy[1] <= 5, zip([1, 4, 5], [3, 2, 5]))))  # Result = [(1, 3)].
	#print('Max =', list(filter(lambda x, y: x <= y, [1, 4, 5], [3, 2, 5])))  # TypeError: filter expected 2 arguments, got 3.
	print('Max =', list(filter(lambda xy: xy[0] <= xy[1], zip([1, 4, 5], [3, 2, 5]))))  # Result = [(1, 3), (5, 5)].

	#--------------------
	items = [3, 4, 5, 6, 7]
	summation = functools.reduce(lambda x, y: x + y, items)
	print('Type of reduce() =', type(summation), summation)  # The type of reduce() is its return type.
	print('Summation =', summation)
	product = functools.reduce(lambda x, y: x * y, items)
	print('Product =', product)

	print('Max =', functools.reduce(max, [5, 8, 3, 1]))
	print('Concatenation =', functools.reduce(lambda s, x: s + str(x), [1, 2, 3, 4], ''))
	print('Flatten =', functools.reduce(operator.concat, [[1, 2], [3, 4], [], [5]], []))

	difference = functools.reduce(lambda x, y: x - y, items, 100)
	print('Difference =', difference)

	# More than two args.
	#print('Max =', functools.reduce(lambda x, y: max(x, y), [5, 8, 3, 1], [2, 5, -1, 7], 0))  # TypeError: reduce expected at most 3 arguments, got 4.
	print('Max =', functools.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])), zip([5, 8, 3, 1], [2, 5, -1, 7])))  # Result = (8, 7).
	print('Min & max =', functools.reduce(lambda x, y: (min(x[0], y[0]), max(x[1], y[1])), zip([5, 8, 3, 1], [2, 5, -1, 7])))  # Result = (1, 7).

	#--------------------
	# Chaining:
	#	Map -> filter -> reduce.
	#	Filter -> map -> reduce.

	print('Chaining =', functools.reduce(lambda x, y: x + y, filter(lambda x: 0 == x % 2, map(lambda x: x**2, range(100)))))

	def evaluate_polynomial(a, x):
		xi = map(lambda i: x**i, range(0, len(a)))  # [x^0, x^1, x^2, ..., x^(n-1)].
		axi = map(operator.mul, a, xi)  # [a[0]*x^0, a[1]*x^1, ..., a[n-1]*x^(n-1)]
		return functools.reduce(operator.add, axi, 0)
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

def inheritance_test():
	class BaseClass(abc.ABC):
		def __init__(self, val):
			self.val = val

		@abc.abstractmethod
		def func1(self, val):
			raise NotImplementedError

		@abc.abstractmethod
		def func2(self, val):
			raise NotImplementedError

		def func3(self, val):
			raise NotImplementedError

	class DerivedClass(BaseClass):
		def __init__(self, val):
			super().__init__(val)

			#self.func1 = self._add
			self.func2 = self._add  # NOTE [caution] >> self._add() is called instead of self._sub().
			self.func3 = self._add

		# The implementation of the abstract method is required.
		def func1(self, val):
			self._sub(val)

		# The implementation of the abstract method is required.
		def func2(self, val):
			self._sub(val)

		# The implementation of the non-abstract method is not required.
		#def func3(self, val):
		#	self._sub(val)

		def _add(self, val):
			print('{} + {} = {}'.format(self.val, val, self.val + val))

		def _sub(self, val):
			print('{} - {} = {}'.format(self.val, val, self.val - val))

	obj = DerivedClass(37)
	obj.func1(17)
	obj.func2(17)
	obj.func3(17)

def main():
	#platform_test()

	#variable_example()

	#assert_test()
	#exception_test()

	#itertools_test()

	#lambda_expression()
	#map_filter_reduce()

	#with_statement_test()

	#caller_func(func)
	#caller_func(func_obj(2))

    #--------------------
    inheritance_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
