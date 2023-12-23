#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, sys, platform, abc, struct
import itertools, functools, operator, difflib
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

def typing_test():
	import typing

	def f(param: None) -> None:
		print(f"param = {param}.")
		return 0

	print(f(None))
	#print(f())  # TypeError: f() missing 1 required positional argument: 'param'.
	print(f(1))

	def f(param: None = None) -> None:
		print(f"param = {param}.")
		return 0

	print(f(None))
	print(f())
	print(f(1))

	def f(param: typing.Optional[int]) -> None:
		print(f"param = {param}.")
		return 0

	print(f(None))
	#print(f())  # TypeError: f() missing 1 required positional argument: 'param'.
	print(f(1))

	def f(param: typing.Optional[int] = None) -> None:
		print(f"param = {param}.")
		return 0

	print(f(None))
	print(f())
	print(f(1))

	def f(a: int, b: int) -> typing.Dict[str, int]:
		return {"a": a, "b": b}
	
	print(f(1, 2))

	def f(a: int, b: int) -> dict[str, int]:
		return {"a": a, "b": b}
	
	print(f(3, 4))

x = 'global'

def variable_test():
	def func1():
		print('x = {}.'.format(x))

	def func2():
		x = x * 2  # UnboundLocalError: local variable 'x' referenced before assignment.
		print('x = {}.'.format(x))

	def func3():
		global x
		x = x * 2
		print('x = {}.'.format(x))

	def func4():
		x = 'local'
		x = x * 2
		print('x = {}.'.format(x))

	func1()
	#func2()
	func3()
	func4()

	#--------------------
	print('globals() = {}.'.format(globals()))

def control_test():
	# For.
	['Even' for val in range(10) if val % 2 == 0]
	#['Even' if val % 2 == 0 for val in range(10)]  # SyntaxError: invalid syntax.

	['Even' if val % 2 == 0 else 'Odd' for val in range(10)]
	#['Even' for val in range(10) if val % 2 == 0 else 'Odd']  # SyntaxError: invalid syntax.

def container_test():
	vals = list(range(5))
	for val in vals:
		# NOTE [caution] >> 3 is not printed.
		print('val = {}.'.format(val))
		if 2 == val:
			vals.remove(val)
	print('vals = {}.'.format(vals))

	vals = list(range(5))
	for idx, val in enumerate(vals):
		# NOTE [caution] >> 3 is not printed.
		print('val = {}.'.format(val))
		if 2 == val:
			del vals[idx]
	print('vals = {}.'.format(vals))

# REF [site] >> https://docs.python.org/3/library/collections.html
def collections_test():
	import collections

	#--------------------
	# collections.namedtuple.

	Point = collections.namedtuple('Point', ['x', 'y'])
	p = Point(11, y=22)

	print('p = {}.'.format(p))
	print('p[0] + p[1] = {}.'.format(p[0] + p[1]))
	print('p.x + p.y = {}.'.format(p.x + p.y))

	x, y = p

	print('Point._make([11, 22]) = {}.'.format(Point._make([11, 22])))
	print('p._asdict() = {}.'.format(p._asdict()))

	# Named tuples are especially useful for assigning field names to result tuples returned by the csv or sqlite3 modules.

	#--------------------
	# collections.OrderedDict.

	rd = {}
	rd['a'] = 'A'
	rd['b'] = 'B'
	rd['c'] = 'C'
	rd['d'] = 'D'
	rd['e'] = 'E'

	for k, v in rd.items():
		print(k, v)

	od = collections.OrderedDict()
	od['a'] = 'A'
	od['b'] = 'B'
	od['c'] = 'C'
	od['d'] = 'D'
	od['e'] = 'E'

	for k, v in od.items():
		print(k, v)

def dataclass_test():
	from dataclasses import dataclass
	import dataclasses

	@dataclass
	class InventoryItem:
		"""Class for keeping track of an item in inventory."""
		name: str
		unit_price: float
		quantity_on_hand: int = 0

		def total_cost(self) -> float:
			return self.unit_price * self.quantity_on_hand

	item = InventoryItem("apple", 2, 3)
	print(f"{item=}.")
	print(f"{dataclasses.asdict(item)=}.")
	print(f"{dataclasses.astuple(item)=}.")
	print(f"{dataclasses.is_dataclass(item)=}.")

	#-----
	@dataclass
	class C:
		mylist: list[int] = dataclasses.field(default_factory=list)

	c = C()
	c.mylist += [1, 2, 3]

	#-----
	@dataclass
	class Point:
		x: int
		y: int

	@dataclass
	class C:
		mylist: list[Point]

	p = Point(10, 20)
	assert dataclasses.asdict(p) == {"x": 10, "y": 20}

	c = C([Point(0, 0), Point(10, 4)])
	assert dataclasses.asdict(c) == {"mylist": [{"x": 0, "y": 0}, {"x": 10, "y": 4}]}

	assert dataclasses.astuple(p) == (10, 20)
	assert dataclasses.astuple(c) == ([(0, 0), (10, 4)],)

def iterable_and_iterator_test():
	# Iterable: an object which one can iterate over.
	#	Sequence: list, string, and tuple.
	#	Others: dictionary, set, file object, and generator.

	numbers = [10, 12, 15, 18, 20]
	fruits = ('apple', 'pineapple', 'blueberry')
	message = 'I love Python'

	#--------------------
	# Iterator: an object which is used to iterate over an iterable object using __next__() method.
	#	An iterator can be created from an iterable by using the function iter().
	#	To make this possible, the class of an iterable needs either a method __iter__, which returns an iterator, or a __getitem__ method with sequential indexes starting with 0.
	#	e.g.) enumerate, zip, reversed, map, filter.

	# Every iterator is also an iterable, but not every iterable is an iterator.

	# Lazy evaluation:
	#	Iterators allow us to both work with and create lazy iterables that don’t do any work until we ask them for their next item.
	#	Because of their laziness, the iterators can help us to deal with infinitely long iterables.

	print('iter(numbers) = {}.'.format(iter(numbers)))
	print('iter(fruits) = {}.'.format(iter(fruits)))
	print('iter(message) = {}.'.format(iter(message)))

	#int_iter = iter(1)  # TypeError: 'int' object is not iterable.

	seq = [10, 20, 30]
	seq_iter = iter(seq)
	try:
		print('next(seq_iter) = {}.'.format(next(seq_iter)))
		print('next(seq_iter) = {}.'.format(next(seq_iter)))
		print('next(seq_iter) = {}.'.format(next(seq_iter)))
		print('next(seq_iter) = {}.'.format(next(seq_iter)))  # StopIteration is raised.
	except StopIteration:
		pass

	for val in seq:
		print('val = {}.'.format(val))
	for val in iter(seq):
		print('val = {}.'.format(val))

	# If we call the iter() function on an iterator it will always give us itself back.
	seq_iter2 = iter(seq_iter)
	print('seq_iter is seq_iter2 = {}.'.format(seq_iter is seq_iter2))

	#print('len(seq_iter) = {}.'.format(len(seq_iter)))  # TypeError: object of type 'list_iterator' has no len().

	#--------------------
	class MyIterable1(object):
		def __init__(self):
			self.seq = list(range(3))

		def __iter__(self):
			print('MyIterable1.__iter__() is called.')
			#return self  # TypeError: iter() returned non-iterator of type 'MyIterable1'.
			#return self.seq  # TypeError: iter() returned non-iterator of type 'list'.
			return iter(self.seq)

		#def __len__(self):
		#	return len(self.seq)

	iterable1 = MyIterable1()
	#print('len(iterable1) = {}.'.format(len(iterable1)))  # TypeError: object of type 'MyIterable1' has no len().
	print('iterable1 = {}.'.format([val for val in iterable1]))
	print('iterable1 = {}.'.format([val for val in iter(iterable1)]))

	class MyIterable2(object):
		def __init__(self):
			self.seq = list(range(3))

		def __getitem__(self, idx):
			print('MyIterable2.__getitem__({}) is called.'.format(idx))
			return self.seq[idx]

		#def __len__(self):
		#	return len(self.seq)

	iterable2 = MyIterable2()
	#print('len(iterable2) = {}.'.format(len(iterable2)))  # TypeError: object of type 'MyIterable2' has no len().
	print('iterable2 = {}.'.format([val for val in iterable2]))
	print('iterable2 = {}.'.format([val for val in iter(iterable2)]))

	class MyIterator(object):
		def __init__(self):
			self.len = 3
			self.seq = list(range(self.len))
			self.index = 0

		def __next__(self):
			print('MyIterator.__next__() is called.')
			if self.index >= self.len: raise StopIteration
			val = self.seq[self.index]
			self.index += 1
			return val

		#def __len__(self):
		#	return self.len

	iterator = MyIterator()
	#print('len(iterator) = {}.'.format(len(iterator)))  # TypeError: object of type 'MyIterator' has no len().
	#print('iterator = {}.'.format([val for val in iterator]))  # TypeError: 'MyIterator' object is not iterable.
	vals = list()
	try:
		for _ in range(100):
			vals.append(next(iterator))
	except StopIteration:
		pass
	print('iterator = {}.'.format(vals))

	class MyNumbers(object):
		def __init__(self):
			self.num = 100  # TODO [check] >> Not applied.

		def __iter__(self):
			print('MyNumbers.__iter__() is called.')
			self.num = 1
			return self

		def __next__(self):
			print('MyNumbers.__next__() is called.')
			val = self.num
			self.num += 1
			return val

		#def __len__(self):
		#	return int(float('inf'))  # OverflowError: cannot convert float infinity to integer.
		#	#import decimal
		#	#return int(decimal.Decimal('Infinity'))  # OverflowError: cannot convert Infinity to integer.

	numbers = MyNumbers()
	#print('len(numbers) = {}.'.format(len(numbers)))  # TypeError: object of type 'MyNumbers' has no len().

	nums = list()
	for num in numbers:
		nums.append(num)
		if num >= 2:
			break
	print('numbers1 = {}.'.format(nums))
	nums = list()
	for num in iter(numbers):
		nums.append(num)
		if num >= 4:
			break
	print('numbers2 = {}.'.format(nums))
	nums = list()
	for num in numbers:
		nums.append(num)
		if num >= 6:
			break
	print('numbers3 = {}.'.format(nums))
	nums = list()
	for _ in range(5):
		nums.append(next(numbers))
	print('numbers4 = {}.'.format(nums))

def assert_test():
	#assert(2 + 2 == 5, 'Error: Addition.')  # Error: Not working.
	assert 2 + 2 == 5, 'Error: Addition.'
	#if not 2 + 2 == 5:
	#	raise AssertionError('Error: Addition')

	if __debug__:  # True if Python is not started with an -O option.
		if not 2 + 2 == 5:
			raise AssertionError
			#raise AssertionError, 'Error: Addition.'  # Error: Invalid syntax.

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
			print('Guard.func() was called: {}, {}.'.format(self._i, d))

	print('Step #1.')
	with Guard(1) as guard:
		print('Step #2.')
		if guard is None:
			print('guard is None.')
		else:
			guard.func(2.0)
		print('Step #3.')
	print('Step #4.')

def function_test():
	# Functions can have attributes ((member) variables).

	def foo():
		foo.var = 10  # No local variable.
		var = 1  # A local variable.
		foo.num_calls += 1
		print(f'id(foo.var) = {id(foo.var)}, id(var) = {id(var)}.')
		print(f'Call {foo.num_calls} of {foo.__name__!r}.')
	foo.num_calls = 0  # Initialization. Similar to static variable in C/C++.

	#print('foo.var = {}.'.format(foo.var))  # AttributeError: 'function' object has no attribute 'var'.

	foo()
	foo()
	foo()

	print('foo.var = {}.'.format(foo.var))

	#--------------------
	# Method and function.
	#	A method in python is somewhat similar to a function, except it is associated with object/classes.

	class MyClass(object):
		def __init__(self):
			self.g = lambda x: x * x

			self.p = self.f
			self.q = self.g

		def f(self, x):
			return x * x

	obj = MyClass()

	print('type(obj.f) = {}.'.format(type(obj.f)))  # <class 'method'>.
	print('type(obj.g) = {}.'.format(type(obj.g)))  # <class 'function'>.
	print('type(obj.p) = {}.'.format(type(obj.p)))  # <class 'method'>.
	print('type(obj.q) = {}.'.format(type(obj.q)))  # <class 'function'>.

	def f1(x):
		return x * x

	f2 = lambda x: x * x

	print('type(f1) = {}.'.format(type(f1)))  # <class 'function'>.
	print('type(f2) = {}.'.format(type(f2)))  # <class 'function'>.

	#--------------------
	# Function signature.
	# "/" indicates that some function parameters must be specified positionally and cannot be used as keyword arguments.
	#	After Python 3.8.
	# "*" forces the caller to use named arguments.

	def func(param1, param2, /, param3, *, param4, param5):
		 print(param1, param2, param3, param4, param5)

	func(10, 20, 30, param4=50, param5=60)
	func(10, 20, param3=30, param4=50, param5=60)

def func(i, f, s):
	print(i, f, s)

class func_obj(object):
	def __init__(self, ii):
		self._ii = ii

	def __call__(self, i, f, s):
		print(i + self._ii, f, s)

def caller_func(callee):
	callee(1, 2.0, 'abc')

def function_call_test():
	caller_func(func)
	caller_func(func_obj(2))

def lambda_expression():
	def make_incrementor(n):
		return lambda x: x + n

	print('Type of lambda expression = {}.'.format(type(make_incrementor)))

	increment_func = make_incrementor(5)
	print(increment_func(1))
	print(increment_func(3))

	a = [1, 2, 3]
	b = [4, 5, 6]
	c = [7, 8, 9]
	a_plus_b_plus_c = list(map(lambda x, y, z: x + y + z, a, b, c))
	print('a + b + c = {}.'.format(a_plus_b_plus_c))

	noop_func = lambda x, y: None
	#def noop_func(x, y): pass
	print('noop_func is called: {}.'.format(noop_func(2, 3)))

	#func2 = lambda x, y: x, y  # NameError: name 'y' is not defined.
	func2 = lambda x, y: (x, y)
	print('func2 is called: {}.'.format(func2(2, 3)))

def map_filter_reduce():
	# Map.

	items = [1, 2, 3, 4, 5]
	squared = map(lambda x: x**2, items)  # class 'map'.
	print('Type of map() = {}: {}.'.format(type(squared), squared))
	print('squared = {}.'.format(list(squared)))

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
	print('Mapping a function with two argments = {}.'.format(list(map_with_two_args)))

	map_with_three_args = map(lambda x, y, z: x + y + z, range(10), range(5), range(8))
	print('Mapping a function with three argments = {}.'.format(list(map_with_three_args)))

	def foo(x):
		if x == 7: raise StopIteration
		elif x < 5: return True
		else: return False

	try:
		vals = [foo(val) for val in range(10)]
	except StopIteration:
		pass
	print('values = {}.'.format(vals)) # NameError: name 'vals' is not defined.
	try:
		vals = map(foo, range(10))
	except StopIteration:
		pass
	print('values = {}.'.format(list(vals)))

	#--------------------
	# Filter.

	number_list = range(-5, 5)
	less_than_zero = filter(lambda x: x < 0, number_list)  # class 'filter'.
	print('Type of filter() = {}: {}.'.format(type(less_than_zero), less_than_zero))
	print('less_than_zero = {}.'.format(list(less_than_zero)))

	# More than two args.
	#print('Max = {}.'.format(list(filter(lambda x, y: x + y <= 5, [1, 4, 5], [3, 2, 5]))))  # TypeError: filter expected 2 arguments, got 3.
	print('Max = {}.'.format(list(filter(lambda xy: xy[0] + xy[1] <= 5, zip([1, 4, 5], [3, 2, 5])))))  # Result = [(1, 3)].
	#print('Max = {}.'.format(list(filter(lambda x, y: x <= y, [1, 4, 5], [3, 2, 5]))))  # TypeError: filter expected 2 arguments, got 3.
	print('Max = {}.'.format(list(filter(lambda xy: xy[0] <= xy[1], zip([1, 4, 5], [3, 2, 5])))))  # Result = [(1, 3), (5, 5)].

	#--------------------
	# Reduce.

	items = [3, 4, 5, 6, 7]
	summation = functools.reduce(lambda x, y: x + y, items)
	print('Type of reduce() = {}: {}.'.format(type(summation), summation))  # The type of reduce() is its return type.
	print('Summation = {}.'.format(summation))
	product = functools.reduce(lambda x, y: x * y, items)
	print('Product = {}.'.format(product))

	print('Max = {}.'.format(functools.reduce(max, [5, 8, 3, 1])))
	print('Concatenation = {}.'.format(functools.reduce(lambda s, x: s + str(x), [1, 2, 3, 4], '')))
	print('Flatten = {}.'.format(functools.reduce(operator.concat, [[1, 2], [3, 4], [], [5]], [])))

	difference = functools.reduce(lambda x, y: x - y, items, 100)
	print('Difference = {}.'.format(difference))

	# More than two args.
	#print('Max = {}.'.format(functools.reduce(lambda x, y: max(x, y), [5, 8, 3, 1], [2, 5, -1, 7], 0)))  # TypeError: reduce expected at most 3 arguments, got 4.
	print('Max = {}.'.format(functools.reduce(lambda x, y: (max(x[0], y[0]), max(x[1], y[1])), zip([5, 8, 3, 1], [2, 5, -1, 7]))))  # Result = (8, 7).
	print('Min & max = {}.'.format(functools.reduce(lambda x, y: (min(x[0], y[0]), max(x[1], y[1])), zip([5, 8, 3, 1], [2, 5, -1, 7]))))  # Result = (1, 7).

	#--------------------
	# Chaining:
	#	Map -> filter -> reduce.
	#	Filter -> map -> reduce.

	print('Chaining = {}.'.format(functools.reduce(lambda x, y: x + y, filter(lambda x: 0 == x % 2, map(lambda x: x**2, range(100))))))

	def evaluate_polynomial(a, x):
		xi = map(lambda i: x**i, range(0, len(a)))  # [x^0, x^1, x^2, ..., x^(n-1)].
		axi = map(operator.mul, a, xi)  # [a[0]*x^0, a[1]*x^1, ..., a[n-1]*x^(n-1)]
		return functools.reduce(operator.add, axi, 0)
	print('Polynomial = {}.'.format(evaluate_polynomial([1, 2, 3, 4], 2)))

# REF [site] >> https://docs.python.org/3/library/itertools.html
def itertools_test():
	print('itertools.count(10) = {}.'.format(type(itertools.count(10))))
	print('itertools.count(10) =', end=' ')
	for item in itertools.count(start=10, step=2):
		print(item, end=', ')
		if item >= 20:
			break
	print()

	#--------------------
	print("itertools.cycle('ABCD') = {}.".format(type(itertools.cycle('ABCD'))))
	print("itertools.cycle('ABCD') =", end=' ')
	for idx, item in enumerate(itertools.cycle('ABCD')):
		print(item, end=', ')
		if idx >= 10:
			break
	print()

	#--------------------
	print('itertools.repeat(37, 7) = {}.'.format(type(itertools.repeat(37, 7))))
	print('itertools.repeat(37, 7) =', end=' ')
	for item in itertools.repeat(37, 7):
		print(item, end=', ')
	print()

	#--------------------
	print('itertools.accumulate([1, 2, 3, 4, 5]) = {}.'.format(type(itertools.accumulate([1, 2, 3, 4, 5]))))
	print('itertools.accumulate([1, 2, 3, 4, 5]) =', end=' ')
	for item in itertools.accumulate([1, 2, 3, 4, 5]):
		print(item, end=', ')
	print()

	#--------------------
	print("itertools.groupby('AAAABBBCCDAABBB') = {}.".format(type(itertools.groupby('AAAABBBCCDAABBB'))))
	print("itertools.groupby('AAAABBBCCDAABBB'): keys = {}.".format(list(k for k, g in itertools.groupby('AAAABBBCCDAABBB'))))
	print("itertools.groupby('AAAABBBCCDAABBB'): groups = {}.".format(list(list(g) for k, g in itertools.groupby('AAAABBBCCDAABBB'))))

	#--------------------
	print("itertools.chain('ABC', 'DEF', 'ghi') = {}.".format(list(itertools.chain('ABC', 'DEF', 'ghi'))))
	print("itertools.chain.from_iterable(['ABC', 'DEF', 'ghi']) = {}.".format(list(itertools.chain.from_iterable(['ABC', 'DEF', 'ghi']))))

	print("itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1]) = {}.".format(list(itertools.compress('ABCDEF', [1, 0, 1, 0, 1, 1]))))
	print("itertools.islice('ABCDEFG', 2, None) = {}.".format(list(itertools.islice('ABCDEFG', 2, None))))

	print('itertools.starmap(pow, [(2, 5), (3, 2), (10, 3)]) = {}.'.format(list(itertools.starmap(pow, [(2, 5), (3, 2), (10, 3)]))))

	#--------------------
	print('itertools.filterfalse(lambda x: x % 2, range(10)) = {}.'.format(list(itertools.filterfalse(lambda x: x % 2, range(10)))))
	print('itertools.dropwhile(lambda x: x < 5, [1, 4, 6, 4, 1] = {}.'.format(list(itertools.dropwhile(lambda x: x < 5, [1, 4, 6, 4, 1]))))
	print('itertools.takewhile(lambda x: x < 5, [1, 4, 6, 4, 1] = {}.'.format(list(itertools.takewhile(lambda x: x < 5, [1, 4, 6, 4, 1]))))

	#--------------------
	print("itertools.zip_longest('ABCD', 'xy', fillvalue='-') = {}.".format(list(itertools.zip_longest('ABCD', 'xy', fillvalue='-'))))

	#--------------------
	print("itertools.tee('ABCDEFG', 2) = {}.".format(itertools.tee('ABCDEFG', 2)))

	#--------------------
	print("itertools.product('ABCD', repeat=2) = {}.".format(list(itertools.product('ABCD', repeat=2))))
	print("itertools.permutations('ABCD', 2) = {}.".format(list(itertools.permutations('ABCD', 2))))
	print("itertools.combinations('ABCD', 2) = {}.".format(list(itertools.combinations('ABCD', 2))))
	print("itertools.combinations_with_replacement('ABCD', 2) = {}.".format(list(itertools.combinations_with_replacement('ABCD', 2))))

# REF [site] >> https://docs.python.org/3/library/difflib.html
def difflib_test():
	"""
	difflib.SequenceMatcher
	difflib.Differ
	difflib.HtmlDiff

	difflib.context_diff()
	difflib.get_close_matches()
	difflib.ndiff()
	difflib.unified_diff()
	"""

	#--------------------
	s = difflib.SequenceMatcher(lambda x: x == ' ', ' abcd', 'abcd abcd')
	print('s.find_longest_match(0, 5, 0, 9) = {}.'.format(s.find_longest_match(0, 5, 0, 9)))

	s = difflib.SequenceMatcher(None, 'abxcd', 'abcd')
	print('s.get_matching_blocks() = {}.'.format(s.get_matching_blocks()))

	a, b = 'qabxcd', 'abycdf'
	s = difflib.SequenceMatcher(None, a, b)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))

	# NOTE [caution] >> The result of a ratio() call may depend on the order of the arguments.
	print("SequenceMatcher(None, 'tide', 'diet').ratio() = {}.".format(difflib.SequenceMatcher(None, 'tide', 'diet').ratio()))
	print("SequenceMatcher(None, 'diet', 'tide').ratio() = {}.".format(difflib.SequenceMatcher(None, 'diet', 'tide').ratio()))

	# Symmetric sequence matching ratio.
	def sequence_matching_ratio(seq1, seq2, isjunk=None):
		"""
		matched_len = 0
		for mb in difflib.SequenceMatcher(isjunk, seq1, seq2).get_matching_blocks():
			matched_len += mb.size
		#return 2 * matched_len, (len(seq1) + len(seq2))
		return 2 * matched_len / (len(seq1) + len(seq2))
		"""
		"""
		matched_len = 0
		for mb in difflib.SequenceMatcher(isjunk, seq1, seq2).get_matching_blocks():
			matched_len += mb.size
		for mb in difflib.SequenceMatcher(isjunk, seq2, seq1).get_matching_blocks():
			matched_len += mb.size
		#return matched_len, (len(seq1) + len(seq2))
		return matched_len / (len(seq1) + len(seq2))
		"""
		#return 2 * functools.reduce(lambda mblen, mb: mblen + mb.size, difflib.SequenceMatcher(isjunk, seq1, seq2).get_matching_blocks(), 0) / (len(seq1) + len(seq2))
		return (functools.reduce(lambda mblen, mb: mblen + mb.size, difflib.SequenceMatcher(isjunk, seq1, seq2).get_matching_blocks(), 0) + functools.reduce(lambda mblen, mb: mblen + mb.size, difflib.SequenceMatcher(isjunk, seq2, seq1).get_matching_blocks(), 0)) / (len(seq1) + len(seq2))

	print("sequence_matching_ratio('tide', 'diet', isjunk=None) = {}.".format(sequence_matching_ratio('tide', 'diet', isjunk=None)))
	print("sequence_matching_ratio('diet', 'tide', isjunk=None) = {}.".format(sequence_matching_ratio('diet', 'tide', isjunk=None)))

	def count_matches(seq1, seq2, isjunk=None):
		return functools.reduce(lambda mblen, mb: mblen + mb.size, difflib.SequenceMatcher(isjunk, seq1, seq2).get_matching_blocks(), 0)

	print("count_matches('tide', 'diet', isjunk=None) = {}.".format(count_matches('tide', 'diet', isjunk=None)))  # 1.
	print("count_matches('diet', 'tide', isjunk=None) = {}.".format(count_matches('diet', 'tide', isjunk=None)))  # 2.

	# The three methods that return the ratio of matching to total characters can give different results due to differing levels of approximation, although quick_ratio() and real_quick_ratio() are always at least as large as ratio():
	s = difflib.SequenceMatcher(None, 'abcd', 'bcde')
	print('s.ratio() = {}.'.format(s.ratio()))  # [0, 1].
	print('s.quick_ratio() = {}.'.format(s.quick_ratio()))  # [0, 1].
	print('s.real_quick_ratio() = {}.'.format(s.real_quick_ratio()))  # [0, 1].

	s = difflib.SequenceMatcher(lambda x: ' ' == x, 'private Thread currentThread;', 'private volatile Thread currentThread;')
	print('round(s.ratio(), 3) = {}.'.format(round(s.ratio(), 3)))
	for block in s.get_matching_blocks():
		print('a[%d] and b[%d] match for %d elements' % block)
	for opcode in s.get_opcodes():
		print('%6s a[%d:%d] b[%d:%d]' % opcode)

	#str1, str2 = '우리나라 대한민국 만세', '대한민국! 우리나라 만.세.'
	str1, str2 = '우리나라 대한민국 만세', '대한민국! 우리 만.세.'
	matcher = difflib.SequenceMatcher(None, str1, str2)  # Long sequence matching. (?)
	#matcher = difflib.SequenceMatcher(lambda x: x == '\n\r', str1, str2)
	#matcher = difflib.SequenceMatcher(lambda x: x == ' \t\n\r', str1, str2)
	print('Ratio = {}.'.format(2 * functools.reduce(lambda mblen, mb: mblen + mb.size, matcher.get_matching_blocks(), 0) / (len(str1) + len(str2))))
	#print('Ratio = {}.'.format(matcher.ratio()))
	for idx, mth in enumerate(matcher.get_matching_blocks()):
		if mth.size != 0:
			print('#{}: {} == {}.'.format(idx, str1[mth.a:mth.a+mth.size], str2[mth.b:mth.b+mth.size]))

	lst1, lst2 = [1, 2, 3, 4, 5], [1, 2, 4, 5]
	matcher = difflib.SequenceMatcher(None, lst1, lst2)
	#matcher = difflib.SequenceMatcher(lambda x: x == '\n\r', str1, str2)
	#matcher = difflib.SequenceMatcher(lambda x: x == ' \t\n\r', str1, str2)
	print('Ratio = {}.'.format(matcher.ratio()))
	for idx, mth in enumerate(matcher.get_matching_blocks()):
		if mth.size != 0:
			print('#{}: {} == {}.'.format(idx, lst1[mth.a:mth.a+mth.size], lst2[mth.b:mth.b+mth.size]))

	# Error case.
	str1, str2 = 'abcabcabcabc', 'abcabdababc'  # Matched sub-sequences = {'abcab', 'ab', 'abc'}.
	matcher = difflib.SequenceMatcher(None, str1, str2)
	for idx, mth in enumerate(matcher.get_matching_blocks()):
		if mth.size != 0:
			print('#{}: {} == {}.'.format(idx, str1[mth.a:mth.a+mth.size], str2[mth.b:mth.b+mth.size]))

	# Other libraries:
	#	https://github.com/jamesturk/jellyfish
	#	https://github.com/life4/textdistance
	#	https://github.com/google/diff-match-patch
	#	https://github.com/mduggan/cdifflib

	#--------------------
	text1 = \
'''  1. Beautiful is better than ugly.
  2. Explicit is better than implicit.
  3. Simple is better than complex.
  4. Complex is better than complicated.
'''.splitlines(keepends=True)
	text2 = \
'''  1. Beautiful is better than ugly.
  3.   Simple is better than complex.
  4. Complicated is better than complex.
  5. Flat is better than nested.
'''.splitlines(keepends=True)

	d = difflib.Differ()
	result = list(d.compare(text1, text2))
	from pprint import pprint
	pprint(result)
	sys.stdout.writelines(result)

	#--------------------
	possibilities = ['hello', 'Hallo', 'hi', 'house', 'key', 'screen', 'hallo', 'question', 'format']
	result = difflib.get_close_matches('Hello', possibilities, n=3, cutoff=0.6)
	print(result)

	import keyword
	result = difflib.get_close_matches('wheel', keyword.kwlist)
	print(result)
	result = difflib.get_close_matches('pineapple', keyword.kwlist)
	print(result)
	result = difflib.get_close_matches('accept', keyword.kwlist)
	print(result)

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

# REF [function] >> struct_test().
def bytes_test():
	# String with encoding 'UTF-8'.
	string = 'Python is interesting.'
	print("bytes(string, 'UTF-8') = {}.".format(bytes(string, 'UTF-8')))

	# Create a byte of given integer size.
	size = 5
	print('bytes(size) = {}.'.format(bytes(size)))

	# Convert iterable list to bytes.
	vals = [1, 2, 3, 4, 5]
	#vals = [1001, 2, 3, 4, 5]  # bytes must be in range(0, 256).
	print('bytes({}) = {}.'.format(vals, bytes(vals)))

	#--------------------
	print("bytes.fromhex('2Ef0 F1f2  ') = {}.".format(bytes.fromhex('2Ef0 F1f2  ')))
	print(r"b'\xf0\xf1\xf2'.hex() = {}.".format(b'\xf0\xf1\xf2'.hex()))

	print(r"b'\xf0\xf1\xf2\xf3\xf4'.hex('-') = {}.".format(b'\xf0\xf1\xf2\xf3\xf4'.hex('-')))
	print(r"b'\xf0\xf1\xf2\xf3\xf4'.hex('_', 2) = {}.".format(b'\xf0\xf1\xf2\xf3\xf4'.hex('_', 2)))
	print(r"b'UUDDLRLRAB'.hex(' ', -4) = {}.".format(b'UUDDLRLRAB'.hex(' ', -4)))

	#--------------------
	# Decoding:
	#	bytes_string.decode(encoding='utf-8', errors='strict')
	#	str(bytes_string, encoding='utf-8', errors='strict')

	print(r"b'Zoot!'.decode() = {}.".format(b'Zoot!'.decode()))
	print(r"b'Zoot!'.decode(encoding='utf-8') = {}.".format(b'Zoot!'.decode(encoding='utf-8')))
	print(r"str(b'Zoot!') = {}.".format(str(b'Zoot!')))
	print(r"str(b'Zoot!', encoding='utf-8') ={}.".format(str(b'Zoot!', encoding='utf-8')))

	# Encoding:
	#	bytes(string, encoding='utf-8', errors='strict')
	#	str.encode(string, encoding='utf-8', errors='strict')

	#print(r"bytes('Zoot!') = {}.".format(bytes('Zoot!'))  # TypeError: string argument without an encoding.
	print(r"bytes('Zoot!', encoding='utf-8') = {}.".format(bytes('Zoot!', encoding='utf-8')))
	print(r"str.encode('Zoot!') = {}.".format(str.encode('Zoot!')))
	print(r"str.encode('Zoot!', encoding='utf-8') = {}.".format(str.encode('Zoot!', encoding='utf-8')))

# REF [site] >> https://docs.python.org/3/library/struct.html
#	struct module performs conversions between Python values and C structs represented as Python bytes objects.
# REF [function] >> bytes_test().
def struct_test():
	#--------------------
	packet = struct.pack('hhl', 1, 2, 3)  # bytes.
	print('Packet = {}.'.format(packet))

	packet1 = struct.unpack('hhl', packet)
	print('Unpacked packet = {}.'.format(packet1))  # tuple: (1, 2, 3).

	#--------------------
	# Endian.

	print('Byte order = {} endian.'.format(sys.byteorder))

	packet = struct.pack('hhl', 1, 2, 3)
	print('Native        = {}.'.format(packet))
	packet = struct.pack('<hhl', 1, 2, 3)  # Little endian.
	print('Little-endian = {}.'.format(packet))
	packet = struct.pack('>hhl', 1, 2, 3)  # Big endian.
	print('Big-endian    = {}.'.format(packet))
	# NOTE [info] >> Native, 
	packet = struct.pack('BLLH', 1, 2, 3, 4)
	print('Native        = {}.'.format(packet))
	packet = struct.pack('<BLLH', 1, 2, 3, 4)  # Little endian.
	print('Little-endian = {}.'.format(packet))
	packet = struct.pack('>BLLH', 1, 2, 3, 4)  # Big endian.
	print('Big-endian    = {}.'.format(packet))

	#--------------------
	record = b'raymond   \x32\x12\x08\x01\x08'
	name, serialnum, school, gradelevel = struct.unpack('<10sHHb', record)
	print(name, serialnum, school, gradelevel)

	# The ordering of format characters may have an impact on size since the padding needed to satisfy alignment requirements is different.
	pack1 = struct.pack('ci', b'*', 0x12131415)
	print(struct.calcsize('ci'), pack1)
	pack2 = struct.pack('ic', 0x12131415, b'*')
	print(struct.calcsize('ic'), pack2)

	# The following format 'llh0l' specifies two pad bytes at the end, assuming longs are aligned on 4-byte boundaries.
	print(struct.pack('llh0l', 1, 2, 3))

def number_system():
	dec_val = 1234
	
	bin_val = 0b101010
	oct_val = 0o76543210
	hex_val = 0x123456789ABCDEF0

	print("bin({}) = '{}'.".format(dec_val, bin(dec_val)))  # String.
	print("oct({}) = '{}'.".format(dec_val, oct(dec_val)))  # String.
	print("hex({}) = '{}'.".format(dec_val, hex(dec_val)))  # String.

	print("int('{}', 2) = {}.".format(bin(dec_val), int(bin(dec_val), 2)))
	print("int('{}', 8) = {}.".format(oct(dec_val), int(oct(dec_val), 8)))
	print("int('{}', 16) = {}.".format(hex(dec_val), int(hex(dec_val), 16)))

def IEEE_754_format():
	# IEEE 754 (binary64) <--> double precision floating-point number.
	ieee754_hex_strs = [
		'3FF0000000000000',  # 1.0.
		'4000000000000000',  # 2.0.
		'C000000000000000',  # -2.0.
		'4008000000000000',  # 3.0.
		'4010000000000000',  # 4.0.
		'4014000000000000',  # 5.0.
		'4018000000000000',  # 6.0.
		'4037000000000000',  # 23.0.
		'3F88000000000000',  # 0.01171875 = 3 / 256.
	]

	for hs in ieee754_hex_strs:
		dbl_val = struct.unpack('<d', struct.pack('<Q', int(hs, 16)))[0]
		hex_val = struct.unpack('<Q', struct.pack('<d', dbl_val))[0]

		print("'{}' (IEEE 754) -> {} (double) -> '{}' (IEEE 754).".format(hs, dbl_val, hex(hex_val)))

def main():
	#platform_test()
	typing_test()

	#variable_test()
	#control_test()
	#container_test()
	#collections_test()
	#dataclass_test()

	#iterable_and_iterator_test()

	#assert_test()
	#exception_test()

	#with_statement_test()

	#function_test()
	#function_call_test()

	#lambda_expression()
	#map_filter_reduce()

	#--------------------
	#itertools_test()
	#difflib_test()

	#--------------------
	#inheritance_test()

	#--------------------
	#bytes_test()
	#struct_test()

	#number_system()  # Binary, octal, decimal, hexadecimal number system.
	#IEEE_754_format()

#--------------------------------------------------------------------

# Usage:
#	python -O
#		__debug__ = False if Python was started with an -O option.
#	python
#		__debug__ = True if Python was not started with an -O option.

if '__main__' == __name__:
	main()
