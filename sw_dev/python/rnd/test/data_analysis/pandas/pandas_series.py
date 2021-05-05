#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np

#---------------------------------------------------------------------

def basic_operation():
	#series = pd.Series([1, 3, 5, np.nan, 6, 8])
	series = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])

	print('series =\n', series, sep='')
	print('a = {}, e = {}'.format(series['a'], series['e']))

def numpy_operation():
	raise NotImplementedError

def main():
	basic_operation()

	#numpy_operation()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
