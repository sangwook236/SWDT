#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.python.org/3/library/typing.html

# NOTE [info] >>
#	The Python runtime does not enforce function and variable type annotations.
#	They can be used by third party tools such as type checkers, IDEs, linters, etc.

#from __future__ import annotations
import typing
import functools, operator

def simple_example():
	user: typing.Tuple[int, str, bool] = (3, "Dale", True)
	nums: typing.List[int] = [1, 2, 3]
	countries: typing.Dict[str, str] = {"KR": "South Korea", "US": "United States", "CN": "China"}

	print("user = {}.".format(user))
	print("nums = {}.".format(nums))
	print("countries = {}.".format(countries))

	TIME_OUT: typing.Final[int] = 10
	TIME_OUT = 20  # NOTE [caution] >> TIME_OUT can be assigned.
	print("TIME_OUT = {}.".format(TIME_OUT))

	#--------------------
	def add(lhs: int, rhs: int) -> int:
		return lhs + rhs

	a, b = 1, 2
	print("add({}, {}) = {}.".format(a, b, add(a, b)))

	#def add_seq(seq: typing.List[int]) -> int:
	#def add_seq(seq: typing.Tuple[int]) -> int:
	#def add_seq(seq: typing.Sequence[int]) -> int:
	def add_seq(seq: typing.Iterable[int]) -> int:
		return sum(seq)

	seq = [1, 2, 3, 4, 5]
	print("add_seq({}) = {}.".format(seq, add_seq(seq)))

	def to_string(num: typing.Union[int, float]) -> str:
		return str(num)

	print("to_string(1) = {}.".format(to_string(1)))
	print("to_string(1.5) = {}.".format(to_string(1.5)))
	print("to_string('abc') = {}.".format(to_string("abc")))  # NOTE [caution] >> A string can be used as an argument.

	def repeat(message: str, times: typing.Optional[int] = None) -> list:
		if times:
			return [message] * times
		else:
			return [message]

	ss, times = "abc", 3
	print("repeat({}) = {}.".format(ss, repeat(ss)))
	print("repeat({}, {}) = {}.".format(ss, times, repeat(ss, times)))

	#--------------------
	def call_func(func: typing.Callable[[typing.Iterable[typing.Any]], typing.Any], args: ...) -> str:
		return func(args)

	def greet(name: str) -> str:
		return "Hello, {} !!!".format(name)

	name = "Sang-Wook"
	print("type(call_func({}, {})) = {}.".format(greet.__name__, name, type(call_func(greet, name))))
	print("call_func({}, {}) = {}.".format(greet.__name__, name, call_func(greet, name)))
	print("call_func({}, {}) = {}.".format(greet.__qualname__, name, call_func(greet, name)))

	def mul(vals: typing.Iterable[float]) -> float:
		return functools.reduce(operator.mul, vals, 1)

	vals = 2, 3
	print("mul({}) = {}.".format(vals, mul(vals)))
	print("type(call_func({}, {})) = {}.".format(mul.__name__, vals, type(call_func(mul, vals))))
	print("call_func({}, {}) = {}.".format(mul.__name__, vals, call_func(mul, vals)))

	#--------------------
	class MyClass(object):
		def __init__(self, val):
			super().__init__()

			self.val: typing.Annotated[int, typing.ValueRange(0, 100)] = val
			self.first_name: typing.Annotated[str, "First Name"]
			self.last_name: typing.Annotated[str, "Last Name"] = "Lee"

	def my_add(lhs: MyClass, rhs: MyClass) -> MyClass:
		return MyClass(lhs.val + rhs.val)

	a, b = MyClass(2), MyClass(3)
	print("my_add({}, {}) = {}.".format(a.val, b.val, my_add(a, b).val))
	#print("a.first_name = {}.".format(a.first_name))  # AttributeError: 'MyClass' object has no attribute 'first_name'.
	print("a.last_name = {}.".format(a.last_name))

	#--------------------
	# Type alias.

	Vector = typing.List[float]

	def scale(scalar: float, vector: Vector) -> Vector:
		return [scalar * num for num in vector]

	scaler = 3
	vals = 2, 3  # NOTE [caution] >> A tuple, not a list.
	print("scale({}, {}) = {}.".format(scaler, vals, scale(scaler, vals)))

	ConnectionOptions = typing.Dict[str, str]
	Address = typing.Tuple[str, int]
	Server = typing.Tuple[Address, ConnectionOptions]

	def broadcast_message(message: str, servers: typing.Sequence[Server]) -> None:
		pass

	#--------------------
	# New type.

	UserId = typing.NewType("UserId", int)
	some_id = UserId(524313)

	def get_user_name(user_id: UserId) -> str:
		pass

	# The static type checker will treat the new type as if it were a subclass of the original type.
	# This is useful in helping catch logical errors.

	# Typechecks.
	user_a = get_user_name(UserId(42351))

	# Does not typecheck; an int is not a UserId.
	user_b = get_user_name(-1)

	#--------------------
	print("typing.get_type_hints(MyClass) = {}.".format(typing.get_type_hints(MyClass)))
	print("typing.get_type_hints(UserId) = {}.".format(typing.get_type_hints(UserId)))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
