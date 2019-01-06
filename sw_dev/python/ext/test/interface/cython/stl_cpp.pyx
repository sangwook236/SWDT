# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector

def vector_test():
	cdef vector[int] vect
	cdef int i, x

	for i in range(10):
		vect.push_back(i)

	for i in range(10):
		print(vect[i])

	for x in vect:
		print(x)

	cdef vector[int] v = range(1, 10, 2)
	print(v)  # [1, 3, 5, 7, 9].

def string_test():
	py_bytes_object = b'The knights who say ni'
	py_unicode_object = u'Those who hear them seldom live to tell the tale.'

	cdef string s = py_bytes_object
	print(s)  # b'The knights who say ni'.

	cdef string cpp_string = <string> py_unicode_object.encode('utf-8')
	print(cpp_string)  # b'Those who hear them seldom live to tell the tale.'

	cdef vector[string] cpp_strings = b'It is a good shrubbery'.split()
	print(cpp_strings[1])  # b'is'.
