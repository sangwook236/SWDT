#!/usr/bin/env python

def fib(n):
	a, b = 0, 1
	while b < n:
		print(b, end = ' ')
		a, b = b, a + b
	print()

def fib2(n):
	result = []
	a, b = 0, 1
	while b < n:
		result.append(b)
		a, b = b, a + b
	return result

def main():
	import sys
	fib(int(sys.argv[1]))

#--------------------------------------------------------------------

# Usage:
#	python module.py 50

if '__main__' == __name__:
	main()
