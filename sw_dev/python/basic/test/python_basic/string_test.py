#!/usr/bin/env python

import string

def basic_operation():
	with open('data.txt', 'r') as myfile:
		data = myfile.read()

	words = data.split()

	with open('data.txt', 'r') as myfile:
		lines = myfile.readlines()

	lines2 = []
	for line in lines:
		lines2.append(line.rstrip('\n'))

	idx1 = data.find('Image loaded:')
	idx2 = data.rfind('Image loaded:')

	import re
	indices = [ss.start() for ss in re.finditer('Image loaded:', data)]

# REF [site] >> https://docs.python.org/3.7/library/string.html
def string_lib_test():
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
	string_lib_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
