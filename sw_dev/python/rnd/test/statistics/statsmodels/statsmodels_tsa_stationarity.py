#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Split time series into two (or more) partitions and compare the mean and variance of each group.
# If they differ and the difference is statistically significant, the time series is likely non-stationary.
def check_statistics(X):
	split = int(len(X) / 2)
	X1, X2 = X[0:split], X[split:]
	mean1, mean2 = X1.mean(), X2.mean()
	var1, var2 = X1.var(), X2.var()
	print('mean1=%f, mean2=%f' % (mean1, mean2))
	print('variance1=%f, variance2=%f' % (var1, var2))

# REF [site] >> https://machinelearningmastery.com/time-series-data-stationary-python/
def simple_approach():
	#series = pd.Series.from_csv('./daily-total-female-births.csv', header=0)
	series = pd.Series.from_csv('./international-airline-passengers.csv', header=0)

	series.hist()
	plt.show()

	check_statistics(series.values)

	#--------------------
	X = np.log(series.values)
	plt.hist(X)
	plt.show()
	plt.plot(X)
	plt.show()

	check_statistics(X)

# The Augmented Dickey-Fuller test:
#	A type of statistical test called a unit root test.
#	REF [site] >> https://en.wikipedia.org/wiki/Unit_root_test
# REF [site] >> https://machinelearningmastery.com/time-series-data-stationary-python/
def augmented_dickey_fuller_test():
	#series = pd.Series.from_csv('./daily-total-female-births.csv', header=0)
	series = pd.Series.from_csv('./international-airline-passengers.csv', header=0)

	result = adfuller(series.values)
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

	#--------------------
	X = np.log(series.values)
	result = adfuller(X)
	print('ADF Statistic: %f' % result[0])
	print('p-value: %f' % result[1])
	print('Critical Values:')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

def main():
	#simple_approach()
	augmented_dickey_fuller_test()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
