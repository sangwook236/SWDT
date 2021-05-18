#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.python.org/3/library/pdb.html

import pdb

def simple_example():
	a = 'aaa'
	pdb.set_trace()
	b = 'bbb'
	c = 'ccc'
	final = a + b + c
	print('final =', final)

def main():
	simple_example()

#--------------------------------------------------------------------

# Usage:
#	python debugging_test.py
#	python -m pdb debugging_test.py

if '__main__' == __name__:
	main()
