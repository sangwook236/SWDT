#!/usr/bin/env python

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

# REF [site] >>
#	https://docs.python.org/3/library/re.html
#	https://docs.python.org/3/howto/regex.html
def regular_expression_example():
	r"""
	# Special sequence.
	\number \A \b \B \d \D \s \S \w \W \Z

	# Standard escape.
	\a \b \f \n \N \r \t \u \U \v \x \\

	# Flag.
	re.A, re.ASCII
	re.I, re.IGNORECASE
	re.L, re.LOCALE
	re.M, re.MULTILINE
	re.S, re.DOTALL
	re.U, re.UNICODE
	re.X, re.VERBOSE

	re.DEBUG

	re.search(pattern, string, flags=0)
	re.match(pattern, string, flags=0)
	re.fullmatch(pattern, string, flags=0)
	re.split(pattern, string, maxsplit=0, flags=0)
	re.findall(pattern, string, flags=0)
	re.finditer(pattern, string, flags=0)
	re.sub(pattern, repl, string, count=0, flags=0)
	re.subn(pattern, repl, string, count=0, flags=0)

	re.escape(pattern)

	re.purge()
	"""

	re.split(r'\W+', 'Words, words, words.')
	re.split(r'(\W+)', 'Words, words, words.')
	re.split(r'\W+', 'Words, words, words.', 1)
	re.split('[a-f]+', '0a3B9', flags=re.IGNORECASE)
	re.split(r'(\W+)', '...words, words...')

	def dash_repl(match):
		if match.group(0) == '-': return ' '
		else: return '-'
	re.sub('-{1,2}', '-', 'pro----gram-files')
	re.sub('-{1,2}', dash_repl, 'pro----gram-files')
	re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)

	re.subn('-{1,2}', dash_repl, 'pro----gram-files')
	re.subn(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE)

	re.escape('http://www.python.org')

	try:
		re.compile('[a-z+')
	except re.error as ex:
		print('re.error: {}.'.format(ex))

	#--------------------
	# Regular expression object.

	"""
	re.compile(pattern, flags=0)

	Pattern.search(string, pos, endpos)
	Pattern.match(string, pos, endpos)
	Pattern.fullmatch(string, pos, endpos)
	Pattern.split(string, maxsplit=0)
	Pattern.findall(string, pos, endpos)
	Pattern.finditer(string, pos, endpos)
	Pattern.sub(repl, string, count=0)
	Pattern.subn(repl, string, count=0)

	Pattern.flags
	Pattern.groups
	Pattern.groupindex
	Pattern.pattern
	"""

	pattern = re.compile('[a-z]+')
	#pattern = re.compile('[a-z]+', re.IGNORECASE)
	#pattern = re.compile(r'\d+\.\d*', re.X)

	pattern = re.compile('d')
	pattern.search('dog')  # Match at index 0.
	pattern.search('dog', 1)  # No match; search doesn't include the 'd'.

	pattern = re.compile("o")
	pattern.match('dog')  # No match as 'o' is not at the start of 'dog'.
	pattern.match('dog', 1)  # Match as 'o' is the 2nd character of 'dog'.

	pattern = re.compile('o[gh]')
	pattern.fullmatch('dog')  # No match as "o" is not at the start of 'dog'.
	pattern.fullmatch('ogre')  # No match as not the full string matches.
	pattern.fullmatch('doggie', 1, 3)  # Matches within given limits.

	pattern = re.compile(r'\d+')
	print(pattern.findall('12 drummers drumming, 11 pipers piping, 10 lords a-leaping'))

	iterator = pattern.finditer('12 drummers drumming, 11 ... 10 ...')
	for match in iterator:
		print(match.span())

	#--------------------
	# Match object.

	"""
	Match.expand()
	Match.group()
	Match.groups()
	Match.groupdict()
	Match.start()
	Match.end()
	Match.span()

	Match.pos
	Match.endpos
	Match.lastindex
	Match.lastgroup
	Match.re
	Match.string
	"""

	pattern = re.compile('[a-z]+')

	print(pattern)
	print(pattern.match(''))

	match = pattern.match('tempo')
	if match:
		print(match)
		print(match.group())
		print(match.start(), match.end())
		print(match.span())
	else:
		print('No match')

	match = re.match(r'(\w+) (\w+)', 'Isaac Newton, physicist')
	match.group(0)  # The entire match.
	match.group(1)  # The first parenthesized subgroup.
	match.group(2)  # The second parenthesized subgroup.
	match.group(1, 2)  # Multiple arguments give us a tuple.

	match = re.match(r'(?P<first_name>\w+) (?P<last_name>\w+)', 'Malcolm Reynolds')
	match.group('first_name')
	match.group('last_name')
	match.group(1)
	match.group(2)

	# If a group matches multiple times, only the last match is accessible.
	match = re.match(r'(..)+', 'a1b2c3')  # Matches 3 times.
	match.group(1)  # Returns only the last match.

	match = re.match(r'(\d+)\.(\d+)', '24.1632')
	match.groups()

	match = re.match(r'(\d+)\.?(\d+)?', '24')
	match.groups()  # Second group defaults to None.
	match.groups('0')  # The second group defaults to '0'.

	match = re.match(r'(?P<first_name>\w+) (?P<last_name>\w+)', 'Malcolm Reynolds')
	match.groupdict()

def main():
	#basic_operation()

	#b_string()
	#u_string()
	#r_string()
	#f_string()

	#string_lib_example()

	#--------------------
	regular_expression_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
