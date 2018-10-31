#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, ConvLSTM2D, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt

# Fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if np.isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

def prepare_dataset():
	# Load all data.
	dataset = pd.read_csv('./household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])

	# Mark all missing values.
	dataset.replace('?', np.nan, inplace=True)
	# Make dataset numeric.
	dataset = dataset.astype(np.float32)

	# Fill missing.
	fill_missing(dataset.values)

	# Add a column for for the remainder of sub metering.
	values = dataset.values
	dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])

	# Save updated dataset
	dataset.to_csv('./household_power_consumption.csv')

def resample_dataset():
	# Load the new file
	dataset = pd.read_csv('./household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Resample minute data to total for each day.
	daily_groups = dataset.resample('D')
	daily_data = daily_groups.sum()

	# summarize.
	print(daily_data.shape)
	print(daily_data.head())

	# Save.
	daily_data.to_csv('./household_power_consumption_days.csv')

# Split a univariate dataset into train/test sets.
def split_dataset(data):
	# Split into standard weeks.
	train, test = data[1:-328], data[-328:-6]
	# Restructure into windows of weekly data.
	train = np.array(np.split(train, len(train) / 7))
	test = np.array(np.split(test, len(test) / 7))
	return train, test

# Evaluate one or more weekly forecasts against expected values.
def evaluate_forecasts(actual, predicted):
	# Calculate an RMSE score for each day.
	scores = list()
	for i in range(actual.shape[1]):
		# Calculate MSE.
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# Calculate RMSE.
		rmse = math.sqrt(mse)
		# Store.
		scores.append(rmse)

	# Calculate overall RMSE.
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))

	return score, scores

# Summarize scores.
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))

# Convert history into inputs and outputs.
def to_supervised(train, n_input, n_out=7):
	# Flatten data.
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))

	X, y = list(), list()
	in_start = 0
	# Step over the entire history one time step at a time.
	for _ in range(len(data)):
		# Define the end of the input sequence.
		in_end = in_start + n_input
		out_end = in_end + n_out
		# Ensure we have enough data for this instance.
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# Move along one time step.
		in_start += 1
	return np.array(X), np.array(y)

# Convert history into inputs and outputs
def to_supervised_for_multivariate(train, n_input, n_out=7):
	# Flatten data.
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
	X, y = list(), list()
	in_start = 0
	# Step over the entire history one time step at a time.
	for _ in range(len(data)):
		# Define the end of the input sequence.
		in_end = in_start + n_input
		out_end = in_end + n_out
		# Ensure we have enough data for this instance.
		if out_end < len(data):
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# Move along one time step.
		in_start += 1
	return np.array(X), np.array(y)

# LSTM model with univariate input and vector output.
def build_univariate_lstm_model(train, n_input):
	# Prepare data.
	train_x, train_y = to_supervised(train, n_input)

	# Define parameters.
	verbose, epochs, batch_size = 0, 70, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Define model.
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')

	# Fit network.
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# Encoder-Decoder LSTM model for multi-step forecasting with univariate input data.
def build_univariate_encdec_lstm_model(train, n_input):
	# Prepare data.
	train_x, train_y = to_supervised(train, n_input)

	# Define parameters.
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Reshape output into [samples, timesteps, features].
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

	# Define model.
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')

	# Fit network.
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# Encoder-Decoder LSTM model for multi-step forecasting with multivariate input data.
def build_multivariate_encdec_lstm_model(train, n_input):
	# Prepare data.
	train_x, train_y = to_supervised_for_multivariate(train, n_input)

	# Define parameters.
	verbose, epochs, batch_size = 0, 50, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Reshape output into [samples, timesteps, features].
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

	# Define model.
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')

	# Fit network.
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model
	pass

# CNN-LSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
def build_univariate_encdec_cnn_lstm_model(train, n_input):
	# Prepare data.
	train_x, train_y = to_supervised(train, n_input)

	# Define parameters.
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Reshape output into [samples, timesteps, features].
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

	# Define model.
	model = Sequential()
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
	model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')

	# Fit network.
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# ConvLSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
def build_univariate_encdec_convlstm_model(train, n_steps, n_length, n_input):
	# Prepare data.
	train_x, train_y = to_supervised(train, n_input)

	# Define parameters.
	verbose, epochs, batch_size = 0, 20, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

	# Reshape into subsequences [samples, time steps, rows, cols, channels].
	train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
	# Reshape output into [samples, timesteps, features].
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

	# Define model.
	model = Sequential()
	model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
	model.add(Flatten())
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')

	# Fit network.
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model

# Make a forecast.
def forecast(model, history, n_input):
	# Flatten data.
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# Retrieve last observations for input data.
	input_x = data[-n_input:, 0]
	# Reshape into [1, n_input, 1].
	input_x = input_x.reshape((1, len(input_x), 1))

	# Forecast the next week.
	yhat = model.predict(input_x, verbose=0)

	# We only want the vector forecast.
	yhat = yhat[0]
	return yhat

# Make a forecast.
def forecast_for_multivariate(model, history, n_input):
	# Flatten data.
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# Retrieve last observations for input data.
	input_x = data[-n_input:, :]
	# Reshape into [1, n_input, n].
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))

	# Forecast the next week.
	yhat = model.predict(input_x, verbose=0)

	# We only want the vector forecast.
	yhat = yhat[0]
	return yhat

# Make a forecast.
def forecast_with_length(model, history, n_steps, n_length, n_input):
	# Flatten data.
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	# Retrieve last observations for input data.
	input_x = data[-n_input:, 0]
	# Reshape into [samples, time steps, rows, cols, channels].
	input_x = input_x.reshape((1, n_steps, 1, n_length, 1))

	# Forecast the next week.
	yhat = model.predict(input_x, verbose=0)

	# We only want the vector forecast.
	yhat = yhat[0]
	return yhat

# Evaluate a single model.
def evaluate_model(build_model_func, forecast_func, train, test, n_input):
	# Fit model.
	model = build_model_func(train, n_input)

	# History is a list of weekly data.
	history = [x for x in train]

	# Walk-forward validation over each week.
	predictions = list()
	for i in range(len(test)):
		# Predict the week.
		yhat_sequence = forecast_func(model, history, n_input)
		# Store the predictions.
		predictions.append(yhat_sequence)
		# Get real observation and add to history for predicting the next week.
		history.append(test[i, :])

	# Evaluate predictions days for each week.
	predictions = np.array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# Evaluate a single model.
def evaluate_model_with_length(build_model_func, forecast_func, train, test, n_steps, n_length, n_input):
	# Fit model.
	model = build_model_func(train, n_steps, n_length, n_input)

	# History is a list of weekly data.
	history = [x for x in train]

	# Walk-forward validation over each week.
	predictions = list()
	for i in range(len(test)):
		# Predict the week.
		yhat_sequence = forecast_func(model, history, n_steps, n_length, n_input)
		# Store the predictions.
		predictions.append(yhat_sequence)
		# Get real observation and add to history for predicting the next week.
		history.append(test[i, :])

	# Evaluate predictions days for each week.
	predictions = np.array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
	return score, scores

# LSTM model with univariate input and vector output.
# REF [site] >> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
def test_univariate_lstm_model():
	# Load the new file.
	dataset = pd.read_csv('./household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Split into train and test.
	train, test = split_dataset(dataset.values)

	# Evaluate model and get scores.
	n_input = 7
	score, scores = evaluate_model(build_univariate_lstm_model, forecast, train, test, n_input)

	# Summarize scores.
	summarize_scores('lstm', score, scores)
	# Plot scores.
	days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
	plt.plot(days, scores, marker='o', label='lstm')
	plt.show()

# Encoder-Decoder LSTM model for multi-step forecasting with univariate input data.
# REF [site] >> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
def test_univariate_encdec_lstm_model():
	# Load the new file.
	dataset = pd.read_csv('./household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Split into train and test.
	train, test = split_dataset(dataset.values)

	# Evaluate model and get scores.
	n_input = 14
	score, scores = evaluate_model(build_univariate_encdec_lstm_model, forecast, train, test, n_input)

	# Summarize scores.
	summarize_scores('lstm', score, scores)
	# Plot scores.
	days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
	plt.plot(days, scores, marker='o', label='lstm')
	plt.show()

# Encoder-Decoder LSTM model for multi-step forecasting with multivariate input data.
# REF [site] >> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
def test_multivariate_encdec_lstm_model():
	# Load the new file.
	dataset = pd.read_csv('./household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Split into train and test.
	train, test = split_dataset(dataset.values)

	# Evaluate model and get scores.
	n_input = 14
	score, scores = evaluate_model(build_multivariate_encdec_lstm_model, forecast_for_multivariate, train, test, n_input)

	# Summarize scores.
	summarize_scores('lstm', score, scores)
	# Plot scores.
	days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
	plt.plot(days, scores, marker='o', label='lstm')
	plt.show()

# CNN-LSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
# REF [site] >> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
def test_univariate_encdec_cnn_lstm_model():
	# Load the new file.
	dataset = pd.read_csv('./household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Split into train and test.
	train, test = split_dataset(dataset.values)

	# Evaluate model and get scores.
	n_input = 14
	score, scores = evaluate_model(build_univariate_encdec_cnn_lstm_model, forecast, train, test, n_input)

	# Summarize scores.
	summarize_scores('lstm', score, scores)
	# Plot scores.
	days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
	plt.plot(days, scores, marker='o', label='lstm')
	plt.show()

# ConvLSTM Encoder-Decoder model for multi-step forecasting with univariate input data.
# REF [site] >> https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
def test_univariate_encdec_convlstm_model():
	# Load the new file.
	dataset = pd.read_csv('./household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

	# Split into train and test.
	train, test = split_dataset(dataset.values)

	# Define the number of subsequences and the length of subsequences.
	n_steps, n_length = 2, 7
	# Define the total days to use as input.
	n_input = n_length * n_steps
	score, scores = evaluate_model_with_length(build_univariate_encdec_convlstm_model, forecast_with_length, train, test, n_steps, n_length, n_input)

	# Summarize scores.
	summarize_scores('lstm', score, scores)
	# Plot scores.
	days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
	plt.plot(days, scores, marker='o', label='lstm')
	plt.show()

def main():
	#prepare_dataset()
	#resample_dataset()

	test_univariate_lstm_model()
	#test_univariate_encdec_lstm_model()
	#test_multivariate_encdec_lstm_model()
	#test_univariate_encdec_cnn_lstm_model()
	#test_univariate_encdec_convlstm_model()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
