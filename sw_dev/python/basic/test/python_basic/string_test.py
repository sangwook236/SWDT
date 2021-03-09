#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import string, re

def basic_operation():
	with open('data.txt', 'r') as myfile:
		data = myfile.read()

	words = data.split()

	with open('data.txt', 'r') as myfile:
		lines = myfile.readlines()

	lines2 = list()
	for line in lines:
		lines2.append(line.rstrip('\n'))

	idx1 = data.find('Image loaded:')
	idx2 = data.rfind('Image loaded:')

	indices = [ss.start() for ss in re.finditer('Image loaded:', data)]

	#--------------------
	# REF [site] >> https://docs.python.org/3/library/stdtypes.html
	"""
	str.capitalize()
	str.casefold()
	str.center()
	str.count()
	str.encode()
	str.endswith()
	str.expandtabs()
	str.find()
	str.format()
	str.format_map()
	str.index()
	str.isalnum()
	str.isalpha()
	str.isascii()
	str.isdecimal()
	str.isdigit()
	str.isidentifier()
	str.islower()
	str.isnumeric()
	str.isprintable()
	str.isspace()
	str.istitle()
	str.isupper()
	str.join()
	str.ljust()
	str.lower()
	str.lstrip()
	str.maketrans()
	str.partition()
	str.replace()
	str.rfind()
	str.rindex()
	str.rjust()
	str.rpartition()
	str.rsplit()
	str.rstrip()
	str.split()
	str.splitlines()
	str.startswith()
	str.strip()
	str.swapcase()
	str.title()
	str.translate()
	str.upper()
	str.zfill()
	"""

	ss = 'my NAME is Sang-Wook.'
	print('ss.capitalize() = {}.'.format(ss.capitalize()))
	print('ss.upper() = {}.'.format(ss.upper()))
	print('ss.lower() = {}.'.format(ss.lower()))
	print('ss.isupper() = {}.'.format(ss.isupper()))
	print('ss.islower() = {}.'.format(ss.islower()))

# r-string: raw string.
#	It ignores escape characters.
#	For example, '\n' is a string containing a newline character, and r'\n' is a string containing a backslash and the letter n.
def r_string():
	print(r"'abc\n123' =", 'abc\n123')
	print(r"r'abc\n123' =", r'abc\n123')

# b-string: binary literal.
#	In Python 2 b-strings do nothing and only exist so that the source code is compatible with Python 3.
#	In Python 3, b-strings allow you to create a bytes object.
def b_string():
	s1 = b'still allows embedded "double" quotes'
	s2 = b"still allows embedded 'single' quotes"
	s3 = b'''3 single quotes'''
	s4 = b"""3 double quotes"""

	# The string must contain two hexadecimal digits per byte, with ASCII whitespace being ignored.
	print("bytes.fromhex('2Ef0 F1f2  ') =", bytes.fromhex('2Ef0 F1f2  '))
	# Return a string object containing two hexadecimal digits for each byte in the instance.
	print(r"b'\xf0\xf1\xf2'.hex() =", b'\xf0\xf1\xf2'.hex())

# u-string: unicode literal.
#	Strings are unicode by default in Python 3, but the 'u' prefix is allowed for backward compatibility with Python 2.
#	In Python 2, strings are ASCII by default, and the 'u' prefix allows you to include non-ASCII characters in your strings.
def u_string():
	s1 = "Fichier non trouvé"
	s2 = u"Fichier non trouvé"
	assert s1 == s2

# f-string:
#	REF [site] >> https://www.python.org/dev/peps/pep-0498/
#	F-strings provide a way to embed expressions inside string literals, using a minimal syntax.
#	It should be noted that an f-string is really an expression evaluated at run time, not a constant value.
def f_string():
	value = 1234
	print(f'The value is {value}.')
	print(f'The value is {value:#06}.')
	print(f'The value is {value:#06x}.')

	value = 12.34
	print(f'The value is {value}.')
	#print(f'The value is {value:#06d}.')  # ValueError.
	print(f'The value is {value:#10.04f}.')
	print(f'The value is {value:#010.04f}.')

	import datetime
	date = datetime.date(1991, 10, 12)
	print(f'{date} was on a {date:%A}')

	# Backslashes may not appear inside the expression portions of f-strings, so you cannot use them, for example, to escape
	#print(f'{\'quoted string\'}')  # SyntaxError.
	print(f'{"quoted string"}')

	print(f'{{ {4*10} }}')
	print(f'{{{4 * 10}}}')

	# Raw f-strings.
	print(fr'x={4*10}\n')
	#print(rf'x={4*10}\n')  # Can be run, but generates a different result.

# REF [site] >> https://docs.python.org/3.7/library/string.html
def string_lib_example():
	print('string.ascii_letters   =', string.ascii_letters)
	print('string.ascii_lowercase =', string.ascii_lowercase)
	print('string.ascii_uppercase =', string.ascii_uppercase)
	print('string.digits          =', string.digits)
	print('string.hexdigits       =', string.hexdigits)
	print('string.octdigits       =', string.octdigits)
	print('string.punctuation     =', string.punctuation)
	print('string.printable       =', string.printable)
	print('string.whitespace      =', string.whitespace)

	#--------------------
	# string.Formatter.

	print('First, thou shalt count to {0}.'.format(1, 2, 3))
	print('Bring me a {}.'.format('cat', 'dog'))
	print('From {} to {}.'.format('a', 'z'))
	print('My age is {age}.'.format(name='abc', age=123))

	class Car(object):
		def __init__(self, weight, height):
			self.weight = weight
			self.height = height

	car = Car(100, 200)
	print('Weight in tons {0.weight}.'.format(car))

	players = ['person1', 'person2', 'person3']
	print('Units destroyed: {players[1]}.'.format(players=players))

def main():
	#basic_operation()

	#b_string()
	#u_string()
	#r_string()
	#f_string()

	string_lib_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
