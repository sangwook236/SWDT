#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

# REF [site] >> https://www.statsmodels.org/stable/index.html
def minimal_example():
	# Load data.
	dat = sm.datasets.get_rdataset('Guerry', 'HistData').data

	# Fit regression model (using the natural log of one of the regressors).
	results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

	# Inspect the results.
	print(results.summary())

	# Generate artificial data (2 regressors + constant).
	num_obs = 100
	X = np.random.random((num_obs, 2))
	X = sm.add_constant(X)

	beta = [1, .1, .5]
	e = np.random.random(num_obs)
	y = np.dot(X, beta) + e

	# Fit regression model.
	results = sm.OLS(y, X).fit()

	# Inspect the results.
	print(results.summary())

# REF [site] >> https://www.statsmodels.org/stable/gettingstarted.html
def getting_started_example():
	df = sm.datasets.get_rdataset('Guerry', 'HistData').data

	vars = ['Department', 'Lottery', 'Literacy', 'Wealth', 'Region']
	df = df[vars]
	print(df[-5:])

	df = df.dropna()
	print(df[-5:])

	y, X = patsy.dmatrices('Lottery ~ Literacy + Wealth + Region', data=df, return_type='dataframe')
	print(y[:3])
	print(X[:3])

	mod = sm.OLS(y, X)  # Describe model.
	res = mod.fit()  # Fit model.

	print(res.summary())  # Summarize model.
	print('res.params =', res.params)
	print('res.rsquared =', res.rsquared)

	# Diagnostics and specification tests.
	#	REF [site] >> https://www.statsmodels.org/stable/stats.html#residual-diagnostics-and-specification-tests

	# Rainbow test for linearity (the null hypothesis is that the relationship is properly modelled as linear).
	print('Rainbow test for linearity =', sm.stats.linear_rainbow(res))
	#print(sm.stats.linear_rainbow.__doc__)

	# Draws a plot of partial regression for a set of regressors.
	sm.graphics.plot_partregress('Lottery', 'Wealth', ['Region', 'Literacy'], data=df, obs_labels=False)

def main():
	#minimal_example()
	getting_started_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
