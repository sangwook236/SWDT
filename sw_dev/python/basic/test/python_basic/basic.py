#!/usr/bin/env python

import os, sys, platform, abc
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

def container_test():
	vals = list(range(5))
	for val in vals:
		# NOTE [caution] >> 3 is not printed.
		print('val = {}.'.format(val))
		if 2 == val:
			vals.remove(val)
	print('vals =', vals)

	vals = list(range(5))
	for idx, val in enumerate(vals):
		# NOTE [caution] >> 3 is not printed.
		print('val = {}.'.format(val))
		if 2 == val:
			del vals[idx]
	print('vals =', vals)

# REF [site] >> https://docs.python.org/3/library/collections.html
def collections_test():
	import collections

	#--------------------
	# collections.namedtuple.

	Point = collections.namedtuple('Point', ['x', 'y'])
	p = Point(11, y=22)

	print('p =', p)
	print('p[0] + p[1] =', p[0] + p[1])
	print('p.x + p.y =', p.x + p.y)

	x, y = p

	print('Point._make([11, 22]) =', Point._make([11, 22]))
	print('p._asdict() =', p._asdict())

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

	print('iter(numbers) =', iter(numbers))
	print('iter(fruits) =', iter(fruits))
	print('iter(message) =', iter(message))

	#int_iter = iter(1)  # TypeError: 'int' object is not iterable.

	seq = [10, 20, 30]
	seq_iter = iter(seq)
	try:
		print('next(seq_iter) =', next(seq_iter))
		print('next(seq_iter) =', next(seq_iter))
		print('next(seq_iter) =', next(seq_iter))
		print('next(seq_iter) =', next(seq_iter))  # StopIteration is raised.
	except StopIteration:
		pass

	for val in seq:
		print('val =', val)
	for val in iter(seq):
		print('val =', val)

	# If we call the iter() function on an iterator it will always give us itself back.
	seq_iter2 = iter(seq_iter)
	print('seq_iter is seq_iter2 =', seq_iter is seq_iter2)

	#print('len(seq_iter) =', len(seq_iter))  # TypeError: object of type 'list_iterator' has no len().

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
	#print('len(iterable1) =', len(iterable1))  # TypeError: object of type 'MyIterable1' has no len().
	print('iterable1 =', [val for val in iterable1])
	print('iterable1 =', [val for val in iter(iterable1)])

	class MyIterable2(object):
		def __init__(self):
			self.seq = list(range(3))

		def __getitem__(self, idx):
			print('MyIterable2.__getitem__({}) is called.'.format(idx))
			return self.seq[idx]

		#def __len__(self):
		#	return len(self.seq)

	iterable2 = MyIterable2()
	#print('len(iterable2) =', len(iterable2))  # TypeError: object of type 'MyIterable2' has no len().
	print('iterable2 =', [val for val in iterable2])
	print('iterable2 =', [val for val in iter(iterable2)])

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
	#print('len(iterator) =', len(iterator))  # TypeError: object of type 'MyIterator' has no len().
	#print('iterator =', [val for val in iterator])  # TypeError: 'MyIterator' object is not iterable.
	vals = list()
	try:
		for _ in range(100):
			vals.append(next(iterator))
	except StopIteration:
		pass
	print('iterator =', vals)

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
	#print('len(numbers) =', len(numbers))  # TypeError: object of type 'MyNumbers' has no len().

	nums = list()
	for num in numbers:
		nums.append(num)
		if num >= 2:
			break
	print('numbers1 =', nums)
	nums = list()
	for num in iter(numbers):
		nums.append(num)
		if num >= 4:
			break
	print('numbers2 =', nums)
	nums = list()
	for num in numbers:
		nums.append(num)
		if num >= 6:
			break
	print('numbers3 =', nums)
	nums = list()
	for _ in range(5):
		nums.append(next(numbers))
	print('numbers4 =', nums)

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

def function_signature_test():
	# "/" indicates that some function parameters must be specified positionally and cannot be used as keyword arguments.
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
	# Map.

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

	def foo(x):
		if x == 7: raise StopIteration
		elif x < 5: return True
		else: return False

	try:
		vals = [foo(val) for val in range(10)]
	except StopIteration:
		pass
	print('values =', vals)  # NameError: name 'vals' is not defined.
	try:
		vals = map(foo, range(10))
	except StopIteration:
		pass
	print('values =', list(vals))

	#--------------------
	# Filter.

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
	# Reduce.

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
	s = difflib.SequenceMatcher(lambda x: x==" ", " abcd", "abcd abcd")
	print('s.find_longest_match(0, 5, 0, 9) =', s.find_longest_match(0, 5, 0, 9))

	s = difflib.SequenceMatcher(None, "abxcd", "abcd")
	print('s.get_matching_blocks() =', s.get_matching_blocks())

	a, b = 'qabxcd', 'abycdf'
	s = difflib.SequenceMatcher(None, a, b)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'.format(tag, i1, i2, j1, j2, a[i1:i2], b[j1:j2]))

	print("SequenceMatcher(None, 'tide', 'diet').ratio() =", difflib.SequenceMatcher(None, 'tide', 'diet').ratio())
	print("SequenceMatcher(None, 'diet', 'tide').ratio() =", difflib.SequenceMatcher(None, 'diet', 'tide').ratio())

	s = difflib.SequenceMatcher(None, 'abcd', 'bcde')
	print('s.ratio() =', s.ratio())
	print('s.quick_ratio() =', s.quick_ratio())
	print('s.real_quick_ratio() =', s.real_quick_ratio())

	s = difflib.SequenceMatcher(lambda x: ' ' == x, 'private Thread currentThread;', 'private volatile Thread currentThread;')
	print('round(s.ratio(), 3) =', round(s.ratio(), 3))
	for block in s.get_matching_blocks():
		print('a[%d] and b[%d] match for %d elements' % block)
	for opcode in s.get_opcodes():
		 print('%6s a[%d:%d] b[%d:%d]' % opcode)

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

# REF [file] >> struct_test.py
def bytes_test():
	# String with encoding 'UTF-8'.
	string = 'Python is interesting.'
	print("bytes(string, 'UTF-8') =", bytes(string, 'UTF-8'))

	# Create a byte of given integer size.
	size = 5
	print('bytes(size) =', bytes(size))

	# Convert iterable list to bytes.
	vals = [1, 2, 3, 4, 5]
	#vals = [1001, 2, 3, 4, 5]  # bytes must be in range(0, 256).
	print('bytes({}) = {}.'.format(vals, bytes(vals)))

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

def main():
	#platform_test()

	#variable_example()
	#container_test()
	#collections_test()

	#iterable_and_iterator_test()

	#assert_test()
	#exception_test()

	#with_statement_test()

	#function_signature_test()
	#function_call_test()

	#lambda_expression()
	map_filter_reduce()

	#--------------------
	#itertools_test()
	#difflib_test()

	#--------------------
	#inheritance_test()

	#--------------------
	#bytes_test()
	#number_system()  # Binary, octal, decimal, hexadecimal number system.

#--------------------------------------------------------------------

# Usage:
#	python -O
#		__debug__ = False if Python was started with an -O option.
#	python
#		__debug__ = True if Python was not started with an -O option.

if '__main__' == __name__:
	main()
