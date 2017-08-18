# REF [site] >> http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# REF [site] >> http://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/

# Unlike regression predictive modeling, time series also adds the complexity of a sequence dependence among the input variables.

#%%-------------------------------------------------------------------

import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%-------------------------------------------------------------------

# Convert an array of values into a dataset matrix.
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back - 1):
		a = dataset[i:(i + look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def display_history(history):
	# List all data in history.
	print(history.history.keys())

	# Summarize history for accuracy.
	fig = plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)
	# Summarize history for loss.
	fig = plt.figure()
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	plt.close(fig)

def plot_graph(dataset, trainPredict, testPredict):
	# Shift train predictions for plotting.
	trainPredictPlot = np.empty_like(dataset)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# Shift test predictions for plotting.
	testPredictPlot = np.empty_like(dataset)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# Plot baseline and predictions.
	plt.plot(scaler.inverse_transform(dataset))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

#%%-------------------------------------------------------------------
# Prepare dataset.

# fix random seed for reproducibility
np.random.seed(7)

# REF [site] >> https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
dataframe = pandas.read_csv('../../../data/machine_learning/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataframe)
plt.show()

dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalize the dataset.
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets.
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

#%%-------------------------------------------------------------------
# LSTM for regression.

def fit_and_predict_lstm_using_window(train, test, look_back, num_epoches, batch_size):
	# Reshape into X=t and Y=t+1.
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of [samples, time steps, features].
	# Our data is in the form [samples, features].

	# Reshape input to be [samples, time steps, features].
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	# Create and fit the LSTM network.
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=num_epoches, batch_size=batch_size, verbose=2)
	display_history(history)

	# Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).

	# Make predictions.
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# Invert predictions.
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY_inv = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY_inv = scaler.inverse_transform([testY])

	# Calculate root mean squared error.
	trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	plot_graph(dataset, trainPredict, testPredict)

look_back = 1
num_epoches = 100
batch_size = 1
fit_and_predict_lstm_using_window(train, test, look_back, num_epoches, batch_size)

#%%-------------------------------------------------------------------
# LSTM for regression using window method.

# Multiple, recent time steps can be used to make the prediction for the next time step.
look_back = 3
num_epoches = 100
batch_size = 1
fit_and_predict_lstm_using_window(train, test, look_back, num_epoches, batch_size)

#%%-------------------------------------------------------------------
# LSTM for regression using time step.

# Some sequence problems may have a varied number of time steps per sample.
# Time steps provide another way to phrase our time series problem.
# Like above in the window example, we can take prior time steps in our time series as inputs to predict the output at the next time step.
# Instead of phrasing the past observations as separate input features, we can use them as time steps of the one input feature, which is indeed a more accurate framing of the problem.

def fit_and_predict_lstm_using_time_step(train, test, look_back, num_epoches, batch_size):
	# Reshape into X=t and Y=t+1.
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of [samples, time steps, features].

	# Reshape input to be [samples, time steps, features].
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# Create and fit the LSTM network.
	model = Sequential()
	model.add(LSTM(4, input_shape=(look_back, 1)))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=num_epoches, batch_size=batch_size, verbose=2)
	display_history(history)

	# Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).

	# Make predictions.
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# Invert predictions.
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY_inv = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY_inv = scaler.inverse_transform([testY])

	# Calculate root mean squared error.
	trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	plot_graph(dataset, trainPredict, testPredict)

look_back = 3
num_epoches = 100
batch_size = 1
fit_and_predict_lstm_using_time_step(train, test, look_back, num_epoches, batch_size)

#%%-------------------------------------------------------------------
# LSTM with memory between batches.

# The LSTM network has memory, which is capable of remembering across long sequences.
# Normally, the state within the network is reset after each training batch when fitting the model, as well as each call to model.predict() or model.evaluate().
# We can gain finer control over when the internal state of the LSTM network is cleared in Keras by making the LSTM layer "stateful".
#	- It requires that the training data not be shuffled when fitting the network.
#	- It also requires explicit resetting of the network state after each exposure to the training data (epoch) by calls to model.reset_states().
#	- This means that we must create our own outer loop of epochs and within each epoch call model.fit() and model.reset_states().
#	- Instead of specifying the input dimensions, we must hard code the number of samples in a batch, number of time steps in a sample and number of features in a time step by setting the batch_input_shape parameter.

def fit_and_predict_lstm_with_memory(train, test, look_back, num_epoches, batch_size):
	# Reshape into X=t and Y=t+1.
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of [samples, time steps, features].

	# Reshape input to be [samples, time steps, features].
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# Create and fit the LSTM network.
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	for epoch in range(num_epoches):
		model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	#display_history(history)

	# Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).

	# Make predictions.
	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)

	# Invert predictions.
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY_inv = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY_inv = scaler.inverse_transform([testY])

	# Calculate root mean squared error.
	trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	plot_graph(dataset, trainPredict, testPredict)

look_back = 3
num_epoches = 100
batch_size = 1
fit_and_predict_lstm_with_memory(train, test, look_back, num_epoches, batch_size)

#%%-------------------------------------------------------------------
# Stacked LSTM with memory between batches.

# LSTM networks can be stacked in Keras in the same way that other layer types can be stacked.
# One addition to the configuration that is required is that an LSTM layer prior to each subsequent LSTM layer must return the sequence.
# This can be done by setting the "return_sequences" parameter on the layer to True.

def fit_and_predict_stacked_lstms_with_memory(train, test, look_back, num_epoches, batch_size):
	# Reshape into X=t and Y=t+1.
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# The LSTM network expects the input data (X) to be provided with a specific array structure in the form of [samples, time steps, features].

	# Reshape input to be [samples, time steps, features].
	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# Create and fit the LSTM network.
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
	model.add(Dense(1))

	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(model.summary())

	for epoch in range(num_epoches):
		model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
		model.reset_states()
	#display_history(history)

	# Note that we invert the predictions before calculating error scores to ensure that performance is reported in the same units as the original data (thousands of passengers per month).

	# Make predictions.
	trainPredict = model.predict(trainX, batch_size=batch_size)
	model.reset_states()
	testPredict = model.predict(testX, batch_size=batch_size)

	# Invert predictions.
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY_inv = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY_inv = scaler.inverse_transform([testY])

	# Calculate root mean squared error.
	trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	plot_graph(dataset, trainPredict, testPredict)

look_back = 3
num_epoches = 100
batch_size = 1
fit_and_predict_stacked_lstms_with_memory(train, test, look_back, num_epoches, batch_size)
