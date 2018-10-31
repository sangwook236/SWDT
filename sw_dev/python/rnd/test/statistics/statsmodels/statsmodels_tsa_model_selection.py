#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings

# One-step naive forecast.
def naive_forecast(history, n):
	return history[-n]

# One-step average forecast.
def average_forecast(history, config):
	n, offset, avg_type = config
	values = list()

	if offset == 1:
		values = history[-n:]
	else:
		# Skip bad configs.
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n, offset))
		# Try and collect n values using offset.
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])

	if avg_type is 'mean':
		# Mean of last n values.
		return np.mean(values)
	else:
		# Median of last n values.
		return np.median(values)

# One-step simple forecast.
def simple_forecast(history, config):
	n, offset, avg_type = config

	# Persist value, ignore other config.
	if avg_type == 'persist':
		return history[-n]

	# Collect values to average.
	values = list()
	if offset == 1:
		values = history[-n:]
	else:
		# Skip bad configs.
		if n*offset > len(history):
			raise Exception('Config beyond end of data: %d %d' % (n, offset))
		# Try and collect n values using offset.
		for i in range(1, n+1):
			ix = i * offset
			values.append(history[-ix])

	# Check if we can average.
	if len(values) < 2:
		raise Exception('Cannot calculate average')

	if avg_type == 'mean':
		# Mean of last n values.
		return np.mean(values)
	else:
		# Median of last n values.
		return np.median(values)

# One-step SARIMA forecast.
def sarima_forecast(history, config):
	order, sorder, trend = config

	# Define model.
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)

	# Fit model.
	model_result = model.fit(disp=False)

	# Make one step forecast.
	pred = model_result.predict(len(history), len(history))
	return pred[0]

# One-step Holt Winter's exponential smoothing forecast.
def exp_smoothing_forecast(history, config):
	t, d, s, p, b, r = config

	# Define model.
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)

	# Fit model.
	model_result = model.fit(optimized=True, use_boxcox=b, remove_bias=r)

	# Make one step forecast.
	pred = model_result.predict(len(history), len(history))
	return pred[0]

# Root mean squared error (RMSE).
def measure_rmse(actual, predicted):
	return math.sqrt(mean_squared_error(actual, predicted))

# Split a univariate dataset into train/test sets.
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# Walk-forward validation for univariate data.
def walk_forward_validation(forecast_func, data, n_test, cfg):
	predictions = list()
	# Split dataset.
	train, test = train_test_split(data, n_test)
	# Seed history with training dataset.
	history = [x for x in train]
	# Step over each time-step in the test set.
	for i in range(len(test)):
		# Fit model and make forecast for history.
		yhat = forecast_func(history, cfg)
		# Store forecast in list of predictions.
		predictions.append(yhat)
		# Add actual observation to history for the next loop.
		history.append(test[i])
	# Estimate prediction error.
	error = measure_rmse(test, predictions)
	return error

# Score a model, return None on failure.
def score_model(forecast_func, data, n_test, cfg, debug=False):
	result = None
	# Convert config to a key.
	key = str(cfg)
	# Show all warnings and fail on exception if debugging.
	if debug:
		result = walk_forward_validation(forecast_func, data, n_test, cfg)
	else:
		# One failure during model validation suggests an unstable config.
		try:
			# Never show warnings when grid searching, too noisy.
			with catch_warnings():
				filterwarnings('ignore')
				result = walk_forward_validation(forecast_func, data, n_test, cfg)
		except:
			result = None

	# Check for an interesting result.
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# Grid search configs.
def grid_search(forecast_func, data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# Execute configs in parallel.
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(forecast_func, data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(forecast_func, data, n_test, cfg) for cfg in cfg_list]

	# Remove empty results.
	scores = [r for r in scores if r[1] != None]
	# Sort configs by error, asc.
	scores.sort(key=lambda tup: tup[1])
	return scores

# Create a set of simple configs to try.
def simple_configs(max_length, offsets=[1]):
	configs = list()
	for i in range(1, max_length+1):
		for o in offsets:
			for t in ['persist', 'mean', 'median']:
				cfg = [i, o, t]
				configs.append(cfg)
	return configs

# Create a set of sarima configs to try.
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [0, 1, 2]
	d_params = [0, 1]
	q_params = [0, 1, 2]
	t_params = ['n', 'c', 't', 'ct']
	P_params = [0, 1, 2]
	D_params = [0, 1]
	Q_params = [0, 1, 2]
	m_params = seasonal
	# Create config instances.
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p, d, q), (P, D, Q, m), t]
									models.append(cfg)
	return models

# Create a set of exponential smoothing configs to try.
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# Create config instances.
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t, d, s, p, b, r]
							models.append(cfg)
	return models

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/
def grid_search_of_naive_method_for_toy_example():
	# Define dataset.
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	# Split data.
	n_test = 4

	"""
	# Test naive forecast.
	for i in range(1, len(data)+1):
		print(naive_forecast(data, i))

	#for i in range(1, len(data)+1):
	for i in range(1, 4):
		print(average_forecast(data, (i, 3, 'mean')))

	#for i in range(1, len(data)+1):
	for i in range(2, 4):
		print(simple_forecast(data, (i, 3, 'mean')))
	"""

	# Model configs.
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)
	#print('cfg_list =', len(cfg_list))

	# Grid search.
	#scores = grid_search(naive_forecast, data, cfg_list, n_test, parallel=True)
	#scores = grid_search(average_forecast, data, cfg_list, n_test, parallel=True)
	scores = grid_search(simple_forecast, data, cfg_list, n_test, parallel=True)
	print('done')
	print('scores =', len(scores))

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/
def grid_search_of_naive_method_without_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('./daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 165

	# Model configs.
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)

	# Grid search.
	scores = grid_search(simple_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/
def grid_search_of_naive_method_with_trend():
	# Parse dates.
	def custom_parser(x):
		return pd.datetime.strptime('195'+x, '%Y-%m')

	# Load dataset.
	series = pd.read_csv('./shampoo.csv', header=0, index_col=0, date_parser=custom_parser)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length)

	# Grid search.
	scores = grid_search(simple_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/
def grid_search_of_naive_method_with_seasonality():
	# Load dataset.
	series = pd.read_csv('./monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length, offsets=[1, 12])

	# Grid search.
	scores = grid_search(simple_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-naive-methods-for-univariate-time-series-forecasting/
def grid_search_of_naive_method_with_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('./monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	max_length = len(data) - n_test
	cfg_list = simple_configs(max_length, offsets=[1, 12])

	# Grid search.
	scores = grid_search(simple_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def grid_search_of_sarima_for_toy_example():
	# Define dataset.
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	#print(data)
	# Split data.
	n_test = 4

	# Model configs.
	cfg_list = sarima_configs()

	# Grid search
	scores = grid_search(sarima_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def grid_search_of_sarima_without_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('./daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 165

	# Model configs.
	cfg_list = sarima_configs()

	# Grid search
	scores = grid_search(sarima_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def grid_search_of_sarima_with_trend():
	# Parse dates.
	def custom_parser(x):
		return pd.datetime.strptime('195'+x, '%Y-%m')

	# Load dataset.
	series = pd.read_csv('./shampoo.csv', header=0, index_col=0, date_parser=custom_parser)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = sarima_configs()

	# Grid search
	scores = grid_search(sarima_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def grid_search_of_sarima_with_seasonality():
	# Load dataset.
	series = pd.read_csv('./monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	# Trim dataset to 5 years.
	data = data[-(5*12):]
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = sarima_configs(seasonal=[0, 12])

	# Grid search
	scores = grid_search(sarima_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
def grid_search_of_sarima_with_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = sarima_configs(seasonal=[0, 6, 12])

	# Grid search
	scores = grid_search(sarima_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
def grid_search_of_exponential_smoothing_for_toy_example():
	# Define dataset.
	data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
	#print(data)
	# Split data.
	n_test = 4

	# Model configs.
	cfg_list = exp_smoothing_configs()

	# Grid search.
	scores = grid_search(exp_smoothing_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
def grid_search_of_exponential_smoothing_without_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('./daily-total-female-births.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 165

	# Model configs.
	cfg_list = exp_smoothing_configs()

	# Grid search.
	scores = grid_search(exp_smoothing_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
def grid_search_of_exponential_smoothing_with_trend():
	# Parse dates.
	def custom_parser(x):
		return pd.datetime.strptime('195'+x, '%Y-%m')

	# Load dataset.
	series = pd.read_csv('./shampoo.csv', header=0, index_col=0, date_parser=custom_parser)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = exp_smoothing_configs()

	# Grid search.
	scores = grid_search(exp_smoothing_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
def grid_search_of_exponential_smoothing_with_seasonality():
	# Load dataset.
	series = pd.read_csv('./monthly-mean-temp.csv', header=0, index_col=0)
	data = series.values
	# Trim dataset to 5 years.
	data = data[-(5*12):]
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = exp_smoothing_configs(seasonal=[12])

	# Grid search.
	scores = grid_search(exp_smoothing_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

# REF [site] >> https://machinelearningmastery.com/how-to-grid-search-triple-exponential-smoothing-for-time-series-forecasting-in-python/
def grid_search_of_exponential_smoothing_with_trend_and_seasonality():
	# Load dataset.
	series = pd.read_csv('./monthly-car-sales.csv', header=0, index_col=0)
	data = series.values
	#print(data)
	# Split data.
	n_test = 12

	# Model configs.
	cfg_list = exp_smoothing_configs(seasonal=[6, 12])

	# Grid search.
	scores = grid_search(exp_smoothing_forecast, data, cfg_list, n_test, parallel=True)
	print('done')

	# List top 3 configs.
	for cfg, error in scores[:3]:
		print(cfg, error)

def main():
	#grid_search_of_naive_method_for_toy_example()
	#grid_search_of_naive_method_without_trend_and_seasonality()
	#grid_search_of_naive_method_with_trend()
	#grid_search_of_naive_method_with_seasonality()
	#grid_search_of_naive_method_with_trend_and_seasonality()

	#grid_search_of_sarima_for_toy_example()
	#grid_search_of_sarima_without_trend_and_seasonality()
	#grid_search_of_sarima_with_trend()
	#grid_search_of_sarima_with_seasonality()
	grid_search_of_sarima_with_trend_and_seasonality()

	#grid_search_of_exponential_smoothing_for_toy_example()
	#grid_search_of_exponential_smoothing_without_trend_and_seasonality()
	#grid_search_of_exponential_smoothing_with_trend()
	#grid_search_of_exponential_smoothing_with_seasonality()
	#grid_search_of_exponential_smoothing_with_trend_and_seasonality()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
