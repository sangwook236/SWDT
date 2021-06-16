#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy.weave

def simple_example():
	weave.inline('std::cout << a << std::endl;', ['a'])
	a = 'string'
	weave.inline('std::cout << a << std::endl;', ['a'])

	sum = np.zeros(3, dtype=np.uint8)
	code = """
		for (int i = 0; i < 10; ++i)
			sum[0] += i;
		for (int i = 0; i < 100; ++i)
			sum[1] += i;
		for (int i = 0; i < 100; ++i)
			sum[2] += i;
	"""
	weave.inline(code, ['sum'])

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
