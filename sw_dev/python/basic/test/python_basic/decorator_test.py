#!/usr/bin/env python

# REF [site] >> https://realpython.com/primer-on-python-decorators/

import math, time, functools
from dataclasses import dataclass
import pint

def simple_function_decorator_example():
	def decorator(func):
		@functools.wraps(func)  # Preserves information about the original function.
		def wrapper():
			print('Decorator is entered.')
			func()
			print('Decorator is exited.')
		return wrapper

	def say_hello():
		print('Hello!')

	say_hello = decorator(say_hello)
	print('Type of decorator =', type(say_hello), say_hello)
	say_hello()

	# NOTE [info] >> Compares to when there is no decorator @functools.wraps(func).
	help(say_hello)

	@decorator
	def say_hello():
		print('Hello!')

	print('Type of decorator =', type(say_hello), say_hello)
	say_hello()

	#--------------------
	def decorator_with_args(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			print('Decorator is entered.')
			func(*args, **kwargs)
			print('Decorator is exited.')
		return wrapper

	@decorator_with_args
	def greet(name):
		print(f'Hello {name}!')

	greet('World')

	#--------------------
	def decorator_with_return(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			print('Decorator is entered.')
			retval = func(*args, **kwargs)
			print('Decorator is exited.')
			return retval
		return wrapper

	@decorator_with_return
	def add_5(rhs):
		return 5 + rhs

	print('5 + 2 =', add_5(2))

def timer(func):
	"""Print the runtime of the decorated function."""
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start_time = time.perf_counter()
		value = func(*args, **kwargs)
		end_time = time.perf_counter()
		run_time = end_time - start_time
		print(f'Finished {func.__name__!r} in {run_time:.4f} secs')
		return value
	return wrapper

def debug(func):
	"""Print the function signature and return value."""
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		args_repr = [repr(a) for a in args]
		kwargs_repr = [f'{k}={v!r}' for k, v in kwargs.items()]
		signature = ', '.join(args_repr + kwargs_repr)
		print(f'Calling {func.__name__}({signature})')
		value = func(*args, **kwargs)
		print(f'{func.__name__!r} returned {value!r}')
		return value
	return wrapper

def function_decorator_example():
	@timer
	def say_hello():
		print('Hello !')

	# Nesting decorators.
	@timer
	@debug
	def greet(name):
		print(f'Hello {name} !')

	PLUGINS = dict()

	# No inner wrapper function.
	def register(func):
		"""Register a function as a plug-in."""
		PLUGINS[func.__name__] = func
		return func

	@register
	def add_5(rhs):
		return 5 + rhs

	print('PLUGINS =', PLUGINS)

	say_hello()
	greet('World')
	print('5 + 2 =', add_5(2))

	#--------------------
	# Decorators with arguments.

	def repeat1(num_times):
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args, **kwargs):
				for _ in range(num_times):
					value = func(*args, **kwargs)
				return value
			return wrapper
		return decorator

	@repeat1(num_times=4)
	def greet1(name):
		print(f'Hello {name} !!')

	greet1('World')

	#--------------------
	# Decorators that can be used both with and without arguments.

	def repeat2(_func=None, *, num_times=2):
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args, **kwargs):
				for _ in range(num_times):
					value = func(*args, **kwargs)
				return value
			return wrapper

		if _func is None:
			return decorator
		else:
			return decorator(_func)

	@repeat2
	def say_hello2():
		print('Hello !!!')

	@repeat2(num_times=4)
	def greet2(name):
		print(f'Hello {name} !!!')

	say_hello2()
	greet2('World')

def class_decorator_example():
	class TimeWaster1:
		@debug
		def __init__(self, max_num):
			self.max_num = max_num

		@timer
		def waste_time(self, num_times):
			for _ in range(num_times):
				sum([i**2 for i in range(self.max_num)])

	tw1 = TimeWaster1(1000)
	tw1.waste_time(999)

	# Writing a class decorator is very similar to writing a function decorator.
	# The only difference is that the decorator will receive a class and not a function as an argument.
	# In fact, function decorators will work as class decorators.
	# When you are using them on a class instead of a function, their effect might not be what you want.

	@timer
	class TimeWaster2:
		def __init__(self, max_num):
			self.max_num = max_num

		def waste_time(self, num_times):
			for _ in range(num_times):
				sum([i**2 for i in range(self.max_num)])

	tw2 = TimeWaster2(1000)
	tw2.waste_time(999)

	#--------------------
	def singleton(cls):
		"""Make a class a Singleton class (only one instance)."""
		@functools.wraps(cls)
		def wrapper(*args, **kwargs):
			if not wrapper.instance:
				wrapper.instance = cls(*args, **kwargs)
			return wrapper.instance
		wrapper.instance = None
		return wrapper

	@singleton
	class Singleton:
		pass

	first = Singleton()
	second = Singleton()

	print('first ID =', id(first))
	print('second ID =', id(second))
	print('first is second ', first is second)

def advanced_decorator_example():
	# Stateful decorators.

	def count_calls(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			wrapper.num_calls += 1
			print(f'Call {wrapper.num_calls} of {func.__name__!r}')
			return func(*args, **kwargs)
		wrapper.num_calls = 0
		return wrapper

	@count_calls
	def say_hello1():
		print('Hello !')

	say_hello1()
	say_hello1()
	
	print('#calls =', say_hello1.num_calls)

	#--------------------
	# Classes as decorators.

	class CountCalls:
		def __init__(self, func):
			functools.update_wrapper(self, func)
			self.func = func
			self.num_calls = 0

		def __call__(self, *args, **kwargs):
			self.num_calls += 1
			print(f'Call {self.num_calls} of {self.func.__name__!r}')
			return self.func(*args, **kwargs)

	@CountCalls
	def say_hello2():
		print('Hello !!')

	say_hello2()
	say_hello2()
	say_hello2()

	#--------------------
	# Caching return values.
	#	Decorators can provide a nice mechanism for caching and memoization.

	@count_calls
	def fibonacci(num):
		if num < 2:
			return num
		return fibonacci(num - 1) + fibonacci(num - 2)

	print('fibonacci(10) =', fibonacci(10))
	print('#calls =', fibonacci.num_calls)

	# In the standard library, a Least Recently Used (LRU) cache is available as @functools.lru_cache.
	# The cache works as a lookup table, so now fibonacci() only does the necessary calculations once:
	def cache(func):
		"""Keep a cache of previous function calls"""
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			cache_key = args + tuple(kwargs.items())
			if cache_key not in wrapper.cache:
				wrapper.cache[cache_key] = func(*args, **kwargs)
			return wrapper.cache[cache_key]
		wrapper.cache = dict()
		return wrapper

	@cache
	@count_calls
	def fibonacci2(num):
		if num < 2:
			return num
		return fibonacci(num - 1) + fibonacci(num - 2)

	# Note that in the final call to fibonacci2(8), no new calculations were needed, since the eighth Fibonacci number had already been calculated for fibonacci2(10).
	fibonacci2(10)
	fibonacci2(8)

	@functools.lru_cache(maxsize=4)
	def fibonacci3(num):
		if num < 2:
			return num
		return fibonacci(num - 1) + fibonacci(num - 2)

	fibonacci3(10)
	fibonacci3(8)

def application():
	# Information about units.
	def set_unit(unit):
		"""Register a unit on a function."""
		def decorator(func):
			func.unit = unit
			return func
		return decorator

	@set_unit('cm^3')
	def volume(radius, height):
		return math.pi * radius**2 * height

	print('volume(3, 5) =', volume(3, 5))
	print('volume.unit =', volume.unit)
	
	# Note that you could have achieved something similar using "function annotations".
	def volume2(radius, height) -> 'cm^3':
		return math.pi * radius**2 * height

	def use_unit(unit):
		"""Have a function return a Quantity with given unit."""
		use_unit.ureg = pint.UnitRegistry()
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args, **kwargs):
				value = func(*args, **kwargs)
				return value * use_unit.ureg(unit)
			return wrapper
		return decorator

	@use_unit('meters per second')
	def average_speed(distance, duration):
		return distance / duration

	bolt = average_speed(100, 9.58)
	print('bolt [meters per second] =', bolt)
	print('bolt [km per hour] =', bolt.to('km per hour'))
	print('bolt [mph] =', bolt.to('mph').m)  # Magnitude.

def main():
	simple_function_decorator_example()
	function_decorator_example()
	class_decorator_example()
	advanced_decorator_example()

	application()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
