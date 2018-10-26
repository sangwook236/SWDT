#!/usr/bin/env python

# REF [site] >> https://www.statsmodels.org/stable/tsa.html

import numpy as np
import pandas as pd
import pandas.util.testing as ptest
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str
import datetime as dt

# REF [site] >> https://www.statsmodels.org/stable/vector_ar.html
def vector_autoregression_example():
	mdata = sm.datasets.macrodata.load_pandas().data

	# Prepare the dates index.
	dates = mdata[['year', 'quarter']].astype(int).astype(str)
	quarterly = dates['year'] + 'Q' + dates['quarter']
	quarterly = dates_from_str(quarterly)
	mdata = mdata[['realgdp', 'realcons', 'realinv']]
	mdata.index = pd.DatetimeIndex(quarterly)
	data = np.log(mdata).diff().dropna()

	# Make a VAR model.
	model = VAR(data)

	results = model.fit(2)
	print(results.summary())

	# Plots input time series.
	results.plot()

	# Plots time series autocorrelation function.
	results.plot_acorr()

	# Lag order selection.
	model.select_order(15)
	results = model.fit(maxlags=15, ic='aic')

	# Forecast.
	lag_order = results.k_ar
	results.forecast(data.values[-lag_order:], 5)

	results.plot_forecast(10)

	# Impulse response analysis.
	# Impulse responses are the estimated responses to a unit impulse in one of the variables.
	# They are computed in practice using the MA(infinity) representation of the VAR(p) process.
	irf = results.irf(10)

	irf.plot(orth=False)
	irf.plot(impulse='realgdp')
	irf.plot_cum_effects(orth=False)

	# Forecast error variance decomposition (FEVD).
	fevd = results.fevd(5)
	print(fevd.summary())

	results.fevd(20).plot()

	# Statistical tests.

	# Granger causality.
	results.test_causality('realgdp', ['realinv', 'realcons'], kind='f')
	# Normality.
	results.test_normality()
	# Whiteness of residuals.
	results.test_whiteness()

# REF [site] >> https://www.statsmodels.org/stable/vector_ar.html
def dynamic_vector_autoregression_example():
	np.random.seed(1)

	ptest.N = 500
	data = ptest.makeTimeDataFrame().cumsum(0)
	#print(data)

	var = DynamicVAR(data, lag_order=2, window_type='expanding')

	print(var.coefs)

	# All estimated coefficients for equation A
	print(var.coefs.minor_xs('A').info())
	# Coefficients on 11/30/2001
	print(var.coefs.major_xs(dt.datetime(2001, 11, 30)).T)

	# Dynamic forecasts for a given number of steps ahead.
	print(var.forecast(2))

	var.plot_forecast(2)

def main():
	vector_autoregression_example()
	dynamic_vector_autoregression_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
