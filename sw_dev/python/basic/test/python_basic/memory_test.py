#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from memory_profiler import profile
import gc

@profile
def func():
	a = [1] * (10 ** 6)
	b = [2] * (2 * 10 ** 7)
	del b
	return a

def main():
	# REF [site] >> https://github.com/pythonprofilers/memory_profiler

	func()

#--------------------------------------------------------------------

# Usage:
#	python -m memory_profiler memory_test.py

if '__main__' == __name__:
	main()
