#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://www.altumintelligence.com/articles/a/Time-Series-Prediction-Using-LSTM-Deep-Neural-Networks
#	https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
#		LSTMs are terrible at time series forecasting.
#	https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#	https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/
#	https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Persistence model:
#	The simplest model that we could use to make predictions would be to persist the last observation.
#	It provides a baseline of performance for the problem that we can use for comparison with an autoregression model.
# REF [site] >> https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def persistence_model():
	# Load dataset.
	def parser(x):
		return pd.datetime.strptime('190'+x, '%Y-%m')
	series = pd.read_csv('./shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

	# Split data into train and test.
	X = series.values
	train, test = X[0:-12], X[-12:]

	# Walk-forward validation.
	history = [x for x in train]
	predictions = list()
	for i in range(len(test)):
		# Make prediction.
		predictions.append(history[-1])
		# Observation.
		history.append(test[i])

	# Report performance.
	rmse = math.sqrt(mean_squared_error(test, predictions))
	print('RMSE: %.3f' % rmse)

	# Line plot of observed vs predicted.
	plt.plot(test)
	plt.plot(predictions)
	plt.show()

# Frame a sequence as a supervised learning problem.
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(lag, 0, -1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# Create a differenced series.
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		diff.append(dataset[i] - dataset[i - interval])
	return np.array(diff)

# Invert differenced value.
def inverse_difference(differenced, dataset):
	inverted = list()
	for (diff, dat) in zip(differenced, dataset):
		inverted.append(diff + dat)
	return np.array(inverted)

# Fit an LSTM network to training data.
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])

	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam')

	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# REF [site] >> https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def data_transformation():
	# Load dataset.
	def parser(x):
		return pd.datetime.strptime('190'+x, '%Y-%m')
	series = pd.read_csv('./shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

	# Transform to be stationary.
	interval = 1
	differenced = difference(series, interval)
	print(differenced)
	# Invert transform.
	inverted = inverse_difference(differenced, series)
	print(inverted)

	# Transform scale.
	X = series.values
	X = X.reshape(len(X), 1)
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(X)
	scaled_X = scaler.transform(X)
	scaled_series = pd.Series(scaled_X[:, 0])
	print(scaled_series.head())
	# Invert transform.
	inverted_X = scaler.inverse_transform(scaled_X)
	inverted_series = pd.Series(inverted_X[:, 0])
	print(inverted_series.head())

	# Transform to supervised learning.
	time_lag = 3
	X = series.values
	supervised = timeseries_to_supervised(X, time_lag)
	print(supervised.head())

# REF [site] >> https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
def univariate_time_series_with_lstm():
	# Load dataset.
	def parser(x):
		return pd.datetime.strptime('190'+x, '%Y-%m')
	series = pd.read_csv('./shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

	# Transform data to be stationary.
	interval = 1
	raw_values = series.values
	diff_values = difference(raw_values, interval)

	# Transform data to be supervised learning.
	time_lag = 1
	supervised = timeseries_to_supervised(diff_values, time_lag)
	supervised_values = supervised.values

	# Split data into train and test sets.
	train, test = supervised_values[0:-12], supervised_values[-12:]
	y_test_true = raw_values[-12:]

	# Transform the scale of the data.
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	train_scaled = scaler.transform(train)
	test_scaled = scaler.transform(test)

	# Repeat experiment.
	num_experiments = 30
	error_scores = list()
	for r in range(num_experiments):
		# Fit the model.
		lstm_model = fit_lstm(train_scaled, 1, 3000, 4)

		# Forecast the entire training dataset to build up state for forecasting.
		train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
		lstm_model.predict(train_reshaped, batch_size=1)

		# Walk-forward validation on the test data.
		# Make one-step forecast.
		X, y = test_scaled[:, 0:-1], test_scaled[:, -1]
		X3 = X.reshape(X.shape[0], 1, X.shape[1])
		yhat = lstm_model.predict(X3, batch_size=1)

		# Invert scaling.
		yhat = scaler.inverse_transform(np.hstack((X, yhat)))
		yhat = yhat[:,-1]
		# Invert differencing.
		yhat = inverse_difference(yhat, raw_values[-12-interval:])

		# Report performance.
		rmse = math.sqrt(mean_squared_error(y_test_true, yhat))
		print('%d) Test RMSE: %.3f' % (r+1, rmse))
		error_scores.append(rmse)

		# Line plot of observed vs predicted.
		#plt.plot(y_test_true)
		#plt.plot(yhat)
		#plt.show()

	# Summarize results.
	results = pd.DataFrame()
	results['rmse'] = error_scores
	print(results.describe())
	results.boxplot()
	plt.show()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# Input sequence (t-n, ..., t-1).
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Forecast sequence (t, t+1, ..., t+n).
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# Put it all together.
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# Drop rows with NaN values.
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# REF [site] >> https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def multivariate_time_series_with_lstm():
	"""
	# Load data.
	def parse(x):
		return pd.datetime.strptime(x, '%Y %m %d %H')
	dataset = pd.read_csv('pollution_raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
	dataset.drop('No', axis=1, inplace=True)

	# Manually specify column names.
	dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
	dataset.index.name = 'date'
	# Mark all NA values with 0.
	dataset['pollution'].fillna(0, inplace=True)
	# Drop the first 24 hours.
	dataset = dataset[24:]

	# Summarize first 5 rows.
	print(dataset.head(5))

	# Save to file.
	dataset.to_csv('./pollution.csv')
	"""

	# Load dataset.
	dataset = pd.read_csv('./pollution.csv', header=0, index_col=0)
	values = dataset.values

	"""
	# Specify columns to plot.
	groups = [0, 1, 2, 3, 5, 6, 7]
	i = 1
	# Plot each column.
	plt.figure()
	for group in groups:
		plt.subplot(len(groups), 1, i)
		plt.plot(values[:, group])
		plt.title(dataset.columns[group], y=0.5, loc='right')
		i += 1
	plt.show()
	"""

	# Integer encode direction.
	encoder = LabelEncoder()
	values[:,4] = encoder.fit_transform(values[:,4])
	# Ensure all data is float.
	values = values.astype('float32')
	# Normalize features.
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaled = scaler.fit_transform(values)
	# Specify the number of lag hours.
	n_hours = 3
	n_features = 8
	# Frame as supervised learning.
	reframed = series_to_supervised(scaled, n_hours, 1)
	# Drop columns we don't want to predict.
	reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
	#print(reframed.head())

	# Split into train and test sets.
	values = reframed.values
	n_train_hours = 365 * 24
	train = values[:n_train_hours, :]
	test = values[n_train_hours:, :]
	# Split into input and outputs.
	n_obs = n_hours * n_features
	train_X, train_y = train[:, :n_obs], train[:, -n_features]
	test_X, test_y = test[:, :n_obs], test[:, -n_features]
	# Reshape input to be 3D [samples, timesteps, features].
	train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
	#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

	# Design network.
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')

	# Fit network.
	history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

	# Plot history.
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()

	# Make a prediction.
	yhat = model.predict(test_X)
	test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

	# Invert scaling for forecast.
	inv_yhat = np.concatenate((yhat, test_X[:, -(n_features-1):]), axis=1)
	inv_yhat = scaler.inverse_transform(inv_yhat)
	inv_yhat = inv_yhat[:,0]
	# Invert scaling for actual.
	test_y = test_y.reshape((len(test_y), 1))
	inv_y = np.concatenate((test_y, test_X[:, -(n_features-1):]), axis=1)
	inv_y = scaler.inverse_transform(inv_y)
	inv_y = inv_y[:,0]

	# Calculate RMSE.
	rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
	print('Test RMSE: %.3f' % rmse)

# Transform series into train and test sets for supervised learning.
def prepare_data(series, n_test, n_lag, n_seq):
	# Extract raw values.
	raw_values = series.values
	raw_values = raw_values.reshape(len(raw_values), 1)
	# Transform into supervised learning problem X, y.
	supervised = series_to_supervised(raw_values, n_lag, n_seq)
	supervised_values = supervised.values
	# Split into train and test sets.
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return train, test

# Make a persistence forecast.
def persistence(last_ob, n_seq):
	return [last_ob for i in range(n_seq)]

# Evaluate the persistence model.
def make_forecasts_for_persistence_model(train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# Make forecast.
		forecast = persistence(X[-1], n_seq)
		# Store the forecast.
		forecasts.append(forecast)
	return forecasts

# Evaluate the RMSE for each forecast time step.
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = test[:,(n_lag+i)]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# Plot the forecasts in the context of the original dataset.
def plot_forecasts(series, forecasts, n_test):
	# Plot the entire dataset in blue.
	plt.plot(series.values)
	# Plot the forecasts in red.
	for i in range(len(forecasts)):
		off_s = len(series) - 12 + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		plt.plot(xaxis, yaxis, color='red')
	# Show the plot.
	plt.show()

# Transform series into train and test sets for supervised learning.
def prepare_data_with_scaler(series, n_test, n_lag, n_seq):
	# Extract raw values.
	raw_values = series.values
	# Transform data to be stationary.
	diff_series = difference(raw_values, 1)
	diff_values = diff_series
	diff_values = diff_values.reshape(len(diff_values), 1)
	# Rescale values to -1, 1.
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# Transform into supervised learning problem X, y.
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# Split into train and test sets.
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test

# Fit an LSTM network to training data.
def fit_lstm_for_multi_step_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# Reshape training into [samples, timesteps, features].
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])

	# Design network.
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')

	# Fit network.
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model

# Make one forecast with an LSTM.
def forecast_lstm(model, X, n_batch):
	# Reshape input pattern to [samples, timesteps, features].
	X = X.reshape(1, 1, len(X))
	# Make forecast.
	forecast = model.predict(X, batch_size=n_batch)
	# Convert to array.
	return [x for x in forecast[0, :]]

# Evaluate the LSTM model.
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# Make forecast.
		forecast = forecast_lstm(model, X, n_batch)
		# Store the forecast.
		forecasts.append(forecast)
	return forecasts

# Invert differenced forecast.
def inverse_difference_for_multi_step_lstm(last_ob, forecast):
	# Invert first forecast.
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# Propagate difference forecast using inverted first value.
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# Inverse data transform on forecasts.
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# Create array from forecast.
		forecast = np.array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# Invert scaling.
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# Invert differencing.
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference_for_multi_step_lstm(last_ob, inv_scale)
		# Store.
		inverted.append(inv_diff)
	return inverted

def evaluate_forecasts_for_multi_step_lstm(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = math.sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

def multi_step_time_series_with_lstm():
	# Load dataset.
	def parser(x):
		return pd.datetime.strptime('190'+x, '%Y-%m')
	series = pd.read_csv('./shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

	"""
	# Summarize first few rows.
	print(series.head())
	# Line plot.
	series.plot()
	plt.show()
	"""

	#--------------------
	# Persistence model.
	n_lag, n_seq, n_test = 1, 3, 10

	# Prepare data.
	train, test = prepare_data(series, n_test, n_lag, n_seq)
	print(test)
	print('Train: %s, Test: %s' % (train.shape, test.shape))

	# Make forecasts.
	forecasts = make_forecasts_for_persistence_model(train, test, n_lag, n_seq)

	# Evaluate forecasts.
	evaluate_forecasts(test, forecasts, n_lag, n_seq)
	# Plot forecasts.
	plot_forecasts(series, forecasts, n_test + 2)

	#--------------------
	n_lag, n_seq, n_test, n_epochs, n_batch, n_neurons = 4, 3, 10, 1500, 1, 1

	# Prepare data.
	scaler, train, test = prepare_data_with_scaler(series, n_test, n_lag, n_seq)

	# Fit model.
	model = fit_lstm_for_multi_step_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

	# Make forecasts.
	forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)

	# Inverse transform forecasts and test.
	forecasts = inverse_transform(series, forecasts, scaler, n_test + 2)
	actual = [row[n_lag:] for row in test]
	actual = inverse_transform(series, actual, scaler, n_test + 2)

	# Evaluate forecasts.
	evaluate_forecasts_for_multi_step_lstm(actual, forecasts, n_lag, n_seq)
	# Plot forecasts.
	plot_forecasts(series, forecasts, n_test + 2)

def main():
	#data_transformation()

	#persistence_model()
	#univariate_time_series_with_lstm()
	#multivariate_time_series_with_lstm()
	multi_step_time_series_with_lstm()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
