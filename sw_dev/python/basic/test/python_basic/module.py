'''
Created on 2009. 2. 22.

@author: sangwook
'''

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

# Usage:
#	python module.py 50

if __name__ == '__main__':
	import sys
	fib(int(sys.argv[1]))
	#pass
