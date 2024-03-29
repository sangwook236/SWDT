#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> 
#	https://www.statsmodels.org/stable/tsa.html
#	https://www.statsmodels.org/stable/statespace.html

import numpy as np
import pandas as pd
import pandas.util.testing as ptest
pd.core.common.is_list_like = pd.api.types.is_list_like
from scipy.stats import norm
from scipy.signal import lfilter
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.base.datetools import dates_from_str
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
from zipfile import ZipFile
from pandas_datareader.data import DataReader
from IPython.display import display, Latex

# REF [site] >> https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
def basic_analysis():
	df = pd.read_csv('./daily-minimum-temperatures-in-me.csv', header=0, index_col=0)
	#df = df[['Temperature']].apply(pd.to_numeric)

	print(df.head())
	df.plot()
	plt.show()

	# Plots the observation at the previous time step t with the observation at the next time step t+1 as a scatter plot.
	#	We can see a large ball of observations along a diagonal line of the plot. It clearly shows a relationship or some correlation.
	pd.plotting.lag_plot(df)
	plt.show()

	# Pairwise correlation.
	values = pd.DataFrame(df.values)
	df2 = pd.concat([values.shift(1), values], axis=1)
	df2.columns = ['t', 't+1']
	result = df2.corr(method='pearson')
	print(result)

	# Autocorrelation plot.
	pd.plotting.autocorrelation_plot(df)
	plt.show()

	plot_acf(df, lags=31)
	plt.show()

# Persistence model:
#	The simplest model that we could use to make predictions would be to persist the last observation.
#	It provides a baseline of performance for the problem that we can use for comparison with an autoregression model.
# REF [site] >> https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
def persistence_model_example():
	df = pd.read_csv('./daily-minimum-temperatures-in-me.csv', header=0, index_col=0)
	#df = df[['Temperature']].apply(pd.to_numeric)

	values = pd.DataFrame(df.values)
	df2 = pd.concat([values.shift(1), values], axis=1)
	df2.columns = ['t', 't+1']

	# Split dataset.
	X = df2.values
	train, test = X[1:len(X)-7], X[len(X)-7:]
	train_X, train_y = train[:,0], train[:,1]
	test_X, test_y = test[:,0], test[:,1]

	# Persistence model.
	def model_persistence(x):
		return x

	# Walk-forward validation.
	predictions = list()
	for x in test_X:
		yhat = model_persistence(x)
		predictions.append(yhat)
	test_score = mean_squared_error(test_y, predictions)
	print('Test MSE: %.3f' % test_score)

	# Plot predictions vs expected.
	plt.plot(test_y)
	plt.plot(predictions, color='red')
	plt.show()

# Autoregression model:
#	A linear regression model that uses lagged variables as input variables.
# REF [site] >> https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
def ar_example():
	df = pd.read_csv('./daily-minimum-temperatures-in-me.csv', header=0, index_col=0)
	#df = df[['Temperature']].apply(pd.to_numeric)

	# Split dataset.
	X = df.values
	train, test = X[1:len(X)-7], X[len(X)-7:]

	# Train autoregression.
	model = AR(train)  # Autoregressive AR(p) model.
	ar_result = model.fit()
	print('Lag: %s' % ar_result.k_ar)
	print('Coefficients: %s' % ar_result.params)

	# Make predictions from fixed AR model.
	predictions = ar_result.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
	for i in range(len(predictions)):
		print('predicted=%f, expected=%f' % (predictions[i], test[i]))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)

	# Plot results.
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()

	# The statsmodels API does not make it easy to update the model as new observations become available.
	# One way would be to re-train the AR model each day as new observations become available, and that may be a valid approach, if not computationally expensive.
	# An alternative would be to use the learned coefficients and manually make predictions. This requires that the history of 29 prior observations be kept and that the coefficients be retrieved from the model and used in the regression equation to come up with new forecasts.

	# Make predictions from rolling AR model.
	window = ar_result.k_ar
	coef = ar_result.params

	# Walk forward over time steps in test.
	history = train[len(train)-window:]
	history = [history[i] for i in range(len(history))]
	predictions = list()
	for t in range(len(test)):
		length = len(history)
		lag = [history[i] for i in range(length-window,length)]
		yhat = coef[0]
		for d in range(window):
			yhat += coef[d+1] * lag[window-d-1]
		obs = test[t]
		predictions.append(yhat)
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)

	# Plot.
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()

# REF [site] >> https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
def arima_example():
	def parser(x):
		return pd.datetime.strptime('190'+x, '%Y-%m')

	series = pd.read_csv('./shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

	print(series.head())
	series.plot()
	plt.show()

	pd.plotting.autocorrelation_plot(series)
	plt.show()

	# Fit model.
	model = ARIMA(series, order=(5,1,0))
	model_result = model.fit(disp=0)
	print(model_result.summary())

	# Plot residual errors.
	# We get a line plot of the residual errors, suggesting that there may still be some trend information not captured by the model.
	residuals = pd.DataFrame(model_result.resid)
	residuals.plot()
	plt.show()
	# The distribution of the residual errors shows that indeed there is a bias in the prediction (a non-zero mean in the residuals).
	residuals.plot(kind='kde')
	plt.show()
	print(residuals.describe())

	# Rolling forecast ARIMA model.
	X = series.values
	size = int(len(X) * 0.66)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=(5, 1, 0))
		model_result = model.fit(disp=0)
		output = model_result.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)

	# Plot.
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()

# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
def sarimax_1_example():
	# Dataset.
	wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
	data = pd.read_stata(BytesIO(wpi1))
	data.index = data.t

	# Fit the model.
	mod = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(1, 1, 1))
	res = mod.fit(disp=False)
	print(res.summary())

# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
def sarimax_2_example():
	# Dataset
	wpi1 = requests.get('http://www.stata-press.com/data/r12/wpi1.dta').content
	data = pd.read_stata(BytesIO(wpi1))
	data.index = data.t
	data['ln_wpi'] = np.log(data['wpi'])
	data['D.ln_wpi'] = data['ln_wpi'].diff()

	# Graph data.
	fig, axes = plt.subplots(1, 2, figsize=(15,4))

	# Levels.
	axes[0].plot(data.index._mpl_repr(), data['wpi'], '-')
	axes[0].set(title='US Wholesale Price Index')

	# Log difference.
	axes[1].plot(data.index._mpl_repr(), data['D.ln_wpi'], '-')
	axes[1].hlines(0, data.index[0], data.index[-1], 'r')
	axes[1].set(title='US Wholesale Price Index - difference of logs');

	# Graph data
	fig, axes = plt.subplots(1, 2, figsize=(15, 4))

	fig = sm.graphics.tsa.plot_acf(data.iloc[1:]['D.ln_wpi'], lags=40, ax=axes[0])
	fig = sm.graphics.tsa.plot_pacf(data.iloc[1:]['D.ln_wpi'], lags=40, ax=axes[1])

	# Fit the model
	#mod = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(1, 1, 4))
	#ar = 1  # The maximum degree specification.
	#ma = (1, 0, 0, 1)  # The lag polynomial specification.
	#mod = sm.tsa.statespace.SARIMAX(data['wpi'], trend='c', order=(ar, 1, ma)))
	mod = sm.tsa.statespace.SARIMAX(data['ln_wpi'], trend='c', order=(1, 1, 1))

	res = mod.fit(disp=False)
	print(res.summary())

# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
def sarimax_3_example():
	# Dataset.
	air2 = requests.get('http://www.stata-press.com/data/r12/air2.dta').content
	data = pd.read_stata(BytesIO(air2))
	data.index = pd.date_range(start=datetime(data.time[0], 1, 1), periods=len(data), freq='MS')
	data['lnair'] = np.log(data['air'])

	# Fit the model.
	mod = sm.tsa.statespace.SARIMAX(data['lnair'], order=(2, 1, 0), seasonal_order=(1, 1, 0, 12), simple_differencing=True)
	res = mod.fit(disp=False)
	print(res.summary())

# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
def sarimax_4_example():
	# Dataset.
	friedman2 = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
	data = pd.read_stata(BytesIO(friedman2))
	data.index = data.time

	# Variables.
	endog = data.loc['1959':'1981', 'consump']
	exog = sm.add_constant(data.loc['1959':'1981', 'm2'])

	# Fit the model.
	mod = sm.tsa.statespace.SARIMAX(endog, exog, order=(1, 0, 1))
	res = mod.fit(disp=False)
	print(res.summary())

# ARIMA postestimation: dynamic forecasting
# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_sarimax_stata.html
def sarimax_postestimation_example():
	# Dataset.
	friedman2 = requests.get('http://www.stata-press.com/data/r12/friedman2.dta').content
	raw = pd.read_stata(BytesIO(friedman2))
	raw.index = raw.time
	data = raw.loc[:'1981']

	# Variables.
	endog = data.loc['1959':, 'consump']
	exog = sm.add_constant(data.loc['1959':, 'm2'])
	nobs = endog.shape[0]

	# Fit the model.
	mod = sm.tsa.statespace.SARIMAX(endog.loc[:'1978-01-01'], exog=exog.loc[:'1978-01-01'], order=(1,0,1))
	fit_res = mod.fit(disp=False)
	print(fit_res.summary())

	mod = sm.tsa.statespace.SARIMAX(endog, exog=exog, order=(1, 0, 1))
	res = mod.filter(fit_res.params)

	# In-sample one-step-ahead predictions.
	predict = res.get_prediction()
	predict_ci = predict.conf_int()

	# Dynamic predictions.
	predict_dy = res.get_prediction(dynamic='1978-01-01')
	predict_dy_ci = predict_dy.conf_int()

	# Graph.
	fig, ax = plt.subplots(figsize=(9,4 ))
	npre = 4
	ax.set(title='Personal consumption', xlabel='Date', ylabel='Billions of dollars')

	# Plot data points.
	data.loc['1977-07-01':, 'consump'].plot(ax=ax, style='o', label='Observed')

	# Plot predictions.
	predict.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='r--', label='One-step-ahead forecast')
	ci = predict_ci.loc['1977-07-01':]
	ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)
	predict_dy.predicted_mean.loc['1977-07-01':].plot(ax=ax, style='g', label='Dynamic forecast (1978)')
	ci = predict_dy_ci.loc['1977-07-01':]
	ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='g', alpha=0.1)

	legend = ax.legend(loc='lower right')

	# Prediction error.

	# Graph.
	fig, ax = plt.subplots(figsize=(9,4))
	npre = 4
	ax.set(title='Forecast error', xlabel='Date', ylabel='Forecast - Actual')

	# In-sample one-step-ahead predictions and 95% confidence intervals.
	predict_error = predict.predicted_mean - endog
	predict_error.loc['1977-10-01':].plot(ax=ax, label='One-step-ahead forecast')
	ci = predict_ci.loc['1977-10-01':].copy()
	ci.iloc[:,0] -= endog.loc['1977-10-01':]
	ci.iloc[:,1] -= endog.loc['1977-10-01':]
	ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.1)

	# Dynamic predictions and 95% confidence intervals.
	predict_dy_error = predict_dy.predicted_mean - endog
	predict_dy_error.loc['1977-10-01':].plot(ax=ax, style='r', label='Dynamic forecast (1978)')
	ci = predict_dy_ci.loc['1977-10-01':].copy()
	ci.iloc[:,0] -= endog.loc['1977-10-01':]
	ci.iloc[:,1] -= endog.loc['1977-10-01':]
	ax.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], color='r', alpha=0.1)

	legend = ax.legend(loc='lower left');
	legend.get_frame().set_facecolor('w')

# REF [site] >> https://machinelearningmastery.com/exponential-smoothing-for-time-series-forecasting-in-python/
def exponential_smoothing_example():
	# Prepare data.
	df = pd.read_csv('./international-airline-passengers.csv', sep=';', parse_dates=['Month'], index_col='Month')
	df.index.freq = 'MS'
	train, test = df.iloc[:130, 0], df.iloc[130:, 0]

	#--------------------
	# Single exponential smoothing, simple exponential smoothing, SES:
	#	A time series forecasting method for univariate data without a trend or seasonality.

	# Create class.
	model = SimpleExpSmoothing(train)

	# Fit model.
	model_result = model.fit()

	# Make prediction.
	pred = model_result.predict(start=test.index[0], end=test.index[-1])

	plt.figure()
	plt.plot(train.index, train, label='Train')
	plt.plot(test.index, test, label='Test')
	plt.plot(pred.index, pred, label='Single Exponential Smoothing (SES)')
	plt.legend(loc='best')
	plt.savefig('ses.jpg')

	#--------------------
	# Double exponential smoothing:
	#	Support for trends in the univariate time series.

	# The method supports trends that change in different ways: an additive and a multiplicative, depending on whether the trend is linear or exponential respectively.
	#	Additive trend: double exponential smoothing with a linear trend.
	#	Multiplicative trend: double exponential smoothing with an exponential trend.
	# Double exponential smoothing with an additive trend: Holt's linear trend model.

	# For longer range (multi-step) forecasts, the trend may continue on unrealistically.
	# As such, it can be useful to dampen the trend over time.
	# Dampening means reducing the size of the trend over future time steps down to a straight line (no trend).

	# As with modeling the trend itself, we can use the same principles in dampening the trend, specifically additively or multiplicatively for a linear or exponential dampening effect.
	#	Additive dampening: dampen a trend linearly.
	#	Multiplicative dampening: dampen the trend exponentially.
	# A damping coefficient is used to control the rate of dampening.

	# Triple exponential smoothing, Holt-Winters exponential smoothing:
	#	Support for seasonality to the univariate time series.

	# As with the trend, the seasonality may be modeled as either an additive or multiplicative process for a linear or exponential change in the seasonality.
	#	Additive seasonality: triple exponential smoothing with a linear seasonality.
	#	Multiplicative seasonality: triple exponential smoothing with an exponential seasonality.

	# Create class.
	model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12)

	# Fit model.
	model_result = model.fit()

	# Make prediction.
	pred = model_result.predict(start=test.index[0], end=test.index[-1])

	plt.figure()
	plt.plot(train.index, train, label='Train')
	plt.plot(test.index, test, label='Test')
	plt.plot(pred.index, pred, label='Holt-Winters')
	plt.legend(loc='best')
	plt.savefig('Holt_Winters.jpg')

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
	print(var.coefs.major_xs(datetime(2001, 11, 30)).T)

	# Dynamic forecasts for a given number of steps ahead.
	print(var.forecast(2))

	var.plot_forecast(2)

# Unobserved components model (also known as "structural time series model").
# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_structural_harvey_jaeger.html
def unobserved_components_model_example():
	# Get the raw data.
	start = '1948-01'
	end = '2008-01'
	us_gnp = DataReader('GNPC96', 'fred', start=start, end=end)
	us_gnp_deflator = DataReader('GNPDEF', 'fred', start=start, end=end)
	us_monetary_base = DataReader('AMBSL', 'fred', start=start, end=end).resample('QS').mean()
	recessions = DataReader('USRECQ', 'fred', start=start, end=end).resample('QS').last().values[:,0]

	# Construct the dataframe.
	dta = pd.concat(map(np.log, (us_gnp, us_gnp_deflator, us_monetary_base)), axis=1)
	dta.columns = ['US GNP','US Prices','US monetary base']
	dates = dta.index._mpl_repr()

	# Plot the data.
	ax = dta.plot(figsize=(13, 3))
	ylim = ax.get_ylim()
	ax.xaxis.grid()
	ax.fill_between(dates, ylim[0]+1e-5, ylim[1]-1e-5, recessions, facecolor='k', alpha=0.1)

	# Model specifications.

	# Unrestricted model, using string specification.
	unrestricted_model = {
		'level': 'local linear trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
	}
	# Unrestricted model, setting components directly.
	# This is an equivalent, but less convenient, way to specify a local linear trend model with a stochastic damped cycle:
	#unrestricted_model = {
	#	'irregular': True, 'level': True, 'stochastic_level': True, 'trend': True, 'stochastic_trend': True,
	#	'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
	#}

	# The restricted model forces a smooth trend.
	restricted_model = {
		'level': 'smooth trend', 'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
	}
	# Restricted model, setting components directly.
	# This is an equivalent, but less convenient, way to specify a smooth trend model with a stochastic damped cycle.
	# Notice that the difference from the local linear trend model is that 'stochastic_level=False' here.
	#unrestricted_model = {
	#	'irregular': True, 'level': True, 'stochastic_level': False, 'trend': True, 'stochastic_trend': True,
	#	'cycle': True, 'damped_cycle': True, 'stochastic_cycle': True
	#}

	# Output.
	output_mod = sm.tsa.UnobservedComponents(dta['US GNP'], **unrestricted_model)
	output_res = output_mod.fit(method='powell', disp=False)

	# Prices.
	prices_mod = sm.tsa.UnobservedComponents(dta['US Prices'], **unrestricted_model)
	prices_res = prices_mod.fit(method='powell', disp=False)

	prices_restricted_mod = sm.tsa.UnobservedComponents(dta['US Prices'], **restricted_model)
	prices_restricted_res = prices_restricted_mod.fit(method='powell', disp=False)

	# Money.
	money_mod = sm.tsa.UnobservedComponents(dta['US monetary base'], **unrestricted_model)
	money_res = money_mod.fit(method='powell', disp=False)

	money_restricted_mod = sm.tsa.UnobservedComponents(dta['US monetary base'], **restricted_model)
	money_restricted_res = money_restricted_mod.fit(method='powell', disp=False)

	print(output_res.summary())

	fig = output_res.plot_components(legend_loc='lower right', figsize=(15, 9));

	# Create Table I.
	table_i = np.zeros((5, 6))

	start = dta.index[0]
	end = dta.index[-1]
	time_range = '%d:%d-%d:%d' % (start.year, start.quarter, end.year, end.quarter)
	models = [
		('US GNP', time_range, 'None'),
		('US Prices', time_range, 'None'),
		('US Prices', time_range, r'$\sigma_\eta^2 = 0$'),
		('US monetary base', time_range, 'None'),
		('US monetary base', time_range, r'$\sigma_\eta^2 = 0$'),
	]
	index = pd.MultiIndex.from_tuples(models, names=['Series', 'Time range', 'Restrictions'])
	parameter_symbols = [
		r'$\sigma_\zeta^2$', r'$\sigma_\eta^2$', r'$\sigma_\kappa^2$', r'$\rho$',
		r'$2 \pi / \lambda_c$', r'$\sigma_\varepsilon^2$',
	]

	i = 0
	for res in (output_res, prices_res, prices_restricted_res, money_res, money_restricted_res):
		if res.model.stochastic_level:
			(sigma_irregular, sigma_level, sigma_trend, sigma_cycle, frequency_cycle, damping_cycle) = res.params
		else:
			(sigma_irregular, sigma_level, sigma_cycle, frequency_cycle, damping_cycle) = res.params
			sigma_trend = np.nan
		period_cycle = 2 * np.pi / frequency_cycle
		
		table_i[i, :] = [
			sigma_level * 1e7, sigma_trend * 1e7,
			sigma_cycle * 1e7, damping_cycle, period_cycle,
			sigma_irregular * 1e7
		]
		i += 1

	pd.set_option('float_format', lambda x: '%.4g' % np.round(x, 2) if not np.isnan(x) else '-')
	table_i = pd.DataFrame(table_i, index=index, columns=parameter_symbols)
	print(table_i)

# VARMAX model.
# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_varmax.html
def varmax_model_example():
	dta = sm.datasets.webuse('lutkepohl2', 'http://www.stata-press.com/data/r12/')
	dta.index = dta.qtr
	endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

	# VAR.
	exog = endog['dln_consump']
	mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(2, 0), trend='nc', exog=exog)
	res = mod.fit(maxiter=1000, disp=False)
	print(res.summary())

	ax = res.impulse_responses(10, orthogonalized=True).plot(figsize=(13, 3))
	ax.set(xlabel='t', title='Responses to a shock to `dln_inv`');

	# VMA.
	mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(0, 2), error_cov_type='diagonal')
	res = mod.fit(maxiter=1000, disp=False)
	print(res.summary())

	# VARMA(p,q) specification.
	mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(1, 1))
	res = mod.fit(maxiter=1000, disp=False)
	print(res.summary())

def compute_coincident_index(mod, res, dta, usphci, dusphci):
	# Estimate W(1).
	spec = res.specification
	design = mod.ssm['design']
	transition = mod.ssm['transition']
	ss_kalman_gain = res.filter_results.kalman_gain[:,:,-1]
	k_states = ss_kalman_gain.shape[0]

	W1 = np.linalg.inv(np.eye(k_states) - np.dot(
		np.eye(k_states) - np.dot(ss_kalman_gain, design),
		transition
	)).dot(ss_kalman_gain)[0]

	# Compute the factor mean vector.
	factor_mean = np.dot(W1, dta.loc['1972-02-01':, 'dln_indprod':'dln_emp'].mean())

	# Normalize the factors.
	factor = res.factors.filtered[0]
	factor *= np.std(usphci.diff()[1:]) / np.std(factor)

	# Compute the coincident index.
	coincident_index = np.zeros(mod.nobs+1)
	# The initial value is arbitrary; here it is set to facilitate comparison.
	coincident_index[0] = usphci.iloc[0] * factor_mean / dusphci.mean()
	for t in range(0, mod.nobs):
		coincident_index[t+1] = coincident_index[t] + factor[t] + factor_mean

	# Attach dates.
	coincident_index = pd.Series(coincident_index, index=dta.index).iloc[1:]

	# Normalize to use the same base year as USPHCI.
	coincident_index *= (usphci.loc['1992-07-01'] / coincident_index.loc['1992-07-01'])

	return coincident_index

class ExtendedDFM(sm.tsa.DynamicFactor):
	def __init__(self, endog, **kwargs):
		# Setup the model as if we had a factor order of 4.
		super(ExtendedDFM, self).__init__(endog, k_factors=1, factor_order=4, error_order=2, **kwargs)

		# Note: 'self.parameters' is an ordered dict with the keys corresponding to parameter types, and the values the number of parameters of that type.
		# Add the new parameters.
		self.parameters['new_loadings'] = 3

		# Cache a slice for the location of the 4 factor AR parameters (a_1, ..., a_4) in the full parameter vector.
		offset = (self.parameters['factor_loadings'] + self.parameters['exog'] + self.parameters['error_cov'])
		self._params_factor_ar = np.s_[offset:offset+2]
		self._params_factor_zero = np.s_[offset+2:offset+4]

	@property
	def start_params(self):
		# Add three new loading parameters to the end of the parameter vector, initialized to zeros (for simplicity; they could be initialized any way you like)
		return np.r_[super(ExtendedDFM, self).start_params, 0, 0, 0]

	@property
	def param_names(self):
		# Add the corresponding names for the new loading parameters (the name can be anything you like).
		return super(ExtendedDFM, self).param_names + ['loading.L%d.f1.%s' % (i, self.endog_names[3]) for i in range(1, 4)]

	def transform_params(self, unconstrained):
		# Perform the typical DFM transformation (w/o the new parameters).
		constrained = super(ExtendedDFM, self).transform_params(unconstrained[:-3])

		# Redo the factor AR constraint, since we only want an AR(2), and the previous constraint was for an AR(4).
		ar_params = unconstrained[self._params_factor_ar]
		constrained[self._params_factor_ar] = (tools.constrain_stationary_univariate(ar_params))

		# Return all the parameters.
		return np.r_[constrained, unconstrained[-3:]]

	def untransform_params(self, constrained):
		# Perform the typical DFM untransformation (w/o the new parameters).
		unconstrained = super(ExtendedDFM, self).untransform_params(constrained[:-3])

		# Redo the factor AR unconstraint, since we only want an AR(2), and the previous unconstraint was for an AR(4).
		ar_params = constrained[self._params_factor_ar]
		unconstrained[self._params_factor_ar] = (tools.unconstrain_stationary_univariate(ar_params))

		# Return all the parameters.
		return np.r_[unconstrained, constrained[-3:]]

	def update(self, params, transformed=True, complex_step=False):
		# Peform the transformation, if required.
		if not transformed:
			params = self.transform_params(params)
		params[self._params_factor_zero] = 0

		# Now perform the usual DFM update, but exclude our new parameters.
		super(ExtendedDFM, self).update(params[:-3], transformed=True, complex_step=complex_step)

		# Finally, set our new parameters in the design matrix.
		self.ssm['design', 3, 1:4] = params[-3:]

# Dynamic factor model.
# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_dfm_coincident.html
def dynamic_factor_model_example():
	np.set_printoptions(precision=4, suppress=True, linewidth=120)

	# Get the datasets from FRED.
	start = '1979-01-01'
	end = '2014-12-01'
	indprod = DataReader('IPMAN', 'fred', start=start, end=end)
	income = DataReader('W875RX1', 'fred', start=start, end=end)
	sales = DataReader('CMRMTSPL', 'fred', start=start, end=end)
	emp = DataReader('PAYEMS', 'fred', start=start, end=end)
	#dta = pd.concat((indprod, income, sales, emp), axis=1)
	#dta.columns = ['indprod', 'income', 'sales', 'emp']

	#HMRMT = DataReader('HMRMT', 'fred', start='1967-01-01', end=end)
	#CMRMT = DataReader('CMRMT', 'fred', start='1997-01-01', end=end)

	#HMRMT_growth = HMRMT.diff() / HMRMT.shift()
	#sales = pd.Series(np.zeros(emp.shape[0]), index=emp.index)

	# Fill in the recent entries (1997 onwards).
	#sales[CMRMT.index] = CMRMT

	# Backfill the previous entries (pre 1997).
	#idx = sales.loc[:'1997-01-01'].index
	#for t in range(len(idx)-1, 0, -1):
	#	month = idx[t]
	#	prev_month = idx[t-1]
	#	sales.loc[prev_month] = sales.loc[month] / (1 + HMRMT_growth.loc[prev_month].values)

	dta = pd.concat((indprod, income, sales, emp), axis=1)
	dta.columns = ['indprod', 'income', 'sales', 'emp']

	dta.loc[:, 'indprod':'emp'].plot(subplots=True, layout=(2, 2), figsize=(15, 6));

	# Create log-differenced series.
	dta['dln_indprod'] = (np.log(dta.indprod)).diff() * 100
	dta['dln_income'] = (np.log(dta.income)).diff() * 100
	dta['dln_sales'] = (np.log(dta.sales)).diff() * 100
	dta['dln_emp'] = (np.log(dta.emp)).diff() * 100

	# De-mean and standardize.
	dta['std_indprod'] = (dta['dln_indprod'] - dta['dln_indprod'].mean()) / dta['dln_indprod'].std()
	dta['std_income'] = (dta['dln_income'] - dta['dln_income'].mean()) / dta['dln_income'].std()
	dta['std_sales'] = (dta['dln_sales'] - dta['dln_sales'].mean()) / dta['dln_sales'].std()
	dta['std_emp'] = (dta['dln_emp'] - dta['dln_emp'].mean()) / dta['dln_emp'].std()

	# Get the endogenous data.
	endog = dta.loc['1979-02-01':, 'std_indprod':'std_emp']

	# Create the model.
	mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=2, error_order=2)
	initial_res = mod.fit(method='powell', disp=False)
	res = mod.fit(initial_res.params, disp=False)

	print(res.summary(separate_params=False))

	# Estimated factors.
	fig, ax = plt.subplots(figsize=(13, 3))

	# Plot the factor.
	dates = endog.index._mpl_repr()
	ax.plot(dates, res.factors.filtered[0], label='Factor')
	ax.legend()

	# Retrieve and also plot the NBER recession indicators.
	rec = DataReader('USREC', 'fred', start=start, end=end)
	ylim = ax.get_ylim()
	ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1)

	# Post-estimation.
	res.plot_coefficients_of_determination(figsize=(8, 2))

	# Coincident index.
	usphci = DataReader('USPHCI', 'fred', start='1979-01-01', end='2014-12-01')['USPHCI']
	usphci.plot(figsize=(13, 3))

	dusphci = usphci.diff()[1:].values

	fig, ax = plt.subplots(figsize=(13, 3))

	# Compute the index.
	coincident_index = compute_coincident_index(mod, res, dta, usphci, dusphci)

	# Plot the factor.
	dates = endog.index._mpl_repr()
	ax.plot(dates, coincident_index, label='Coincident index')
	ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
	ax.legend(loc='lower right')

	# Retrieve and also plot the NBER recession indicators.
	ylim = ax.get_ylim()
	ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1)

	# Extended dynamic factor model.

	# Create the model.
	extended_mod = ExtendedDFM(endog)
	initial_extended_res = extended_mod.fit(maxiter=1000, disp=False)
	extended_res = extended_mod.fit(initial_extended_res.params, method='nm', maxiter=1000)
	print(extended_res.summary(separate_params=False))

	extended_res.plot_coefficients_of_determination(figsize=(8, 2))

	fig, ax = plt.subplots(figsize=(13,3))

	# Compute the index.
	extended_coincident_index = compute_coincident_index(extended_mod, extended_res, dta, usphci, dusphci)

	# Plot the factor.
	dates = endog.index._mpl_repr()
	ax.plot(dates, coincident_index, '-', linewidth=1, label='Basic model')
	ax.plot(dates, extended_coincident_index, '--', linewidth=3, label='Extended model')
	ax.plot(usphci.index._mpl_repr(), usphci, label='USPHCI')
	ax.legend(loc='lower right')
	ax.set(title='Coincident indices, comparison')

	# Retrieve and also plot the NBER recession indicators.
	ylim = ax.get_ylim()
	ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1)

# Custom state space model.
# REF [site] >> https://www.statsmodels.org/stable/statespace.html
def custom_state_space_model_example():
	# True model parameters.
	nobs = int(1e3)
	true_phi = np.r_[0.5, -0.2]
	true_sigma = 1**0.5

	# Simulate a time series.
	np.random.seed(1234)
	disturbances = np.random.normal(0, true_sigma, size=(nobs,))
	endog = lfilter([1], np.r_[1, -true_phi], disturbances)

	# Construct the model.
	class AR2(sm.tsa.statespace.MLEModel):
		def __init__(self, endog):
			# Initialize the state space model
			super(AR2, self).__init__(endog, k_states=2, k_posdef=1, initialization='stationary')

			# Setup the fixed components of the state space representation.
			self['design'] = [1, 0]
			self['transition'] = [[0, 0], [1, 0]]
			self['selection', 0, 0] = 1

		# Describe how parameters enter the model.
		def update(self, params, transformed=True, **kwargs):
			params = super(AR2, self).update(params, transformed, **kwargs)

			self['transition', 0, :] = params[:2]
			self['state_cov', 0, 0] = params[2]

		# Specify start parameters and parameter names.
		@property
		def start_params(self):
			return [0, 0, 1]  # These are very simple.

	# Create and fit the model
	mod = AR2(endog)
	res = mod.fit()
	print(res.summary())

# Univariate Local Linear Trend Model.
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
	def __init__(self, endog):
		# Model order.
		k_states = k_posdef = 2

		# Initialize the statespace.
		super(LocalLinearTrend, self).__init__(
			endog, k_states=k_states, k_posdef=k_posdef,
			initialization='approximate_diffuse',
			loglikelihood_burn=k_states
		)

		# Initialize the matrices.
		self.ssm['design'] = np.array([1, 0])
		self.ssm['transition'] = np.array([[1, 1], [0, 1]])
		self.ssm['selection'] = np.eye(k_states)

		# Cache some indices.
		self._state_cov_idx = ('state_cov',) + np.diag_indices(k_posdef)

	@property
	def param_names(self):
		return ['sigma2.measurement', 'sigma2.level', 'sigma2.trend']

	@property
	def start_params(self):
		return [np.std(self.endog)] * 3

	def transform_params(self, unconstrained):
		return unconstrained**2

	def untransform_params(self, constrained):
		return constrained**0.5

	def update(self, params, *args, **kwargs):
		params = super(LocalLinearTrend, self).update(params, *args, **kwargs)

		# Observation covariance.
		self.ssm['obs_cov',0,0] = params[0]

		# State covariance.
		self.ssm[self._state_cov_idx] = params[1:]

# State space modeling: Local linear trend.
# REF [site] >> https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_local_linear_trend.html
def state_space_modeling_example():
	# Download the dataset.
	ck = requests.get('http://staff.feweb.vu.nl/koopman/projects/ckbook/OxCodeAll.zip').content
	zipped = ZipFile(BytesIO(ck))
	df = pd.read_table(
		BytesIO(zipped.read('OxCodeIntroStateSpaceBook/Chapter_2/NorwayFinland.txt')),
		skiprows=1, header=None, sep='\s+', engine='python',
		names=['date', 'nf', 'ff']
	)

	# Load Dataset.
	df.index = pd.date_range(start='%d-01-01' % df.date[0], end='%d-01-01' % df.iloc[-1, 0], freq='AS')

	# Log transform.
	df['lff'] = np.log(df['ff'])

	# Setup the model.
	mod = LocalLinearTrend(df['lff'])

	# Fit it using MLE (recall that we are fitting the three variance parameters).
	res = mod.fit(disp=False)
	print(res.summary())

	# Perform prediction and forecasting.
	predict = res.get_prediction()
	forecast = res.get_forecast('2014')

	fig, ax = plt.subplots(figsize=(10, 4))

	# Plot the results.
	df['lff'].plot(ax=ax, style='k.', label='Observations')
	predict.predicted_mean.plot(ax=ax, label='One-step-ahead Prediction')
	predict_ci = predict.conf_int(alpha=0.05)
	predict_index = np.arange(len(predict_ci))
	ax.fill_between(predict_index[2:], predict_ci.iloc[2:, 0], predict_ci.iloc[2:, 1], alpha=0.1)

	forecast.predicted_mean.plot(ax=ax, style='r', label='Forecast')
	forecast_ci = forecast.conf_int()
	forecast_index = np.arange(len(predict_ci), len(predict_ci) + len(forecast_ci))
	ax.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], alpha=0.1)

	# Cleanup the image.
	ax.set_ylim((4, 8));
	legend = ax.legend(loc='lower left');

def main():
	#basic_analysis()

	#persistence_model_example()
	#ar_example()
	#arima_example()

	#sarimax_1_example()
	#sarimax_2_example()
	#sarimax_3_example()
	#sarimax_4_example()
	sarimax_postestimation_example()

	#exponential_smoothing_example()

	#vector_autoregression_example()
	#dynamic_vector_autoregression_example()

	#unobserved_components_model_example()
	#varmax_model_example()
	#dynamic_factor_model_example()

	#custom_state_space_model_example()
	#state_space_modeling_example()

	# Model selection.
	# REF [file] >> statsmodels_tsa_model_selection.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
