#!/usr/bin/env python

import pandas as pd
import numpy as np

#%%-------------------------------------------------------------------

def basic_operation():
	series = pd.Series([1, 3, 5, np.nan, 6, 8])
	print('series =\n', series, sep='')

def numpy_operation():
	raise NotImplementedError

def main():
	basic_operation()

	#numpy_operation()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
