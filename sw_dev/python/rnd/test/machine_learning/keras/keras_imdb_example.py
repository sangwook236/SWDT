# REF [site] >> http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

#%%-------------------------------------------------------------------

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb

#%%-------------------------------------------------------------------

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

def load_dataset(max_features, max_len):
	(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	print(len(x_train), 'train sequences')
	print(len(x_test), 'test sequences')
    
	# Pad sequences (samples x time).
	x_train = sequence.pad_sequences(x_train, maxlen=max_len)
	x_test = sequence.pad_sequences(x_test, maxlen=max_len)
	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)

	return x_train, y_train, x_test, y_test

#%%-------------------------------------------------------------------
# LSTM.

max_features = 20000
maxlen = 80  # Cut texts after this number of words (among top max_features most common words).
batch_size = 32
num_epoches = 15

# Loading data.
x_train, y_train, x_test, y_test = load_dataset(max_features, maxlen)

# Build model.
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Try using different optimizers and different optimizer configs.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epoches, validation_data=(x_test, y_test))
display_history(history)

# Evaluate.
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#%%-------------------------------------------------------------------
# Bidirectional LSTM.

max_features = 20000
# Cut texts after this number of words (among top max_features most common words).
maxlen = 100
batch_size = 32
num_epochs = 4

# Loading data.
x_train, y_train, x_test, y_test = load_dataset(max_features, maxlen)

# Build model.
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Try using different optimizers and different optimizer configs.
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Train.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=[x_test, y_test])
display_history(history)

# Evaluate.
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#%%-------------------------------------------------------------------
# CNN.

# Set parameters.
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
num_epochs = 2

# Loading data.
x_train, y_train, x_test, y_test = load_dataset(max_features, maxlen)

# Build model.
model = Sequential()

# Start off with an efficient embedding layer which maps our vocab indices into embedding_dims dimensions.
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.2))

# Add a Convolution1D, which will learn filters word group filters of size filter_length.
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
# Use max pooling.
model.add(GlobalMaxPooling1D())

# Add a vanilla hidden layer.
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# Project onto a single unit output layer, and squash it with a sigmoid.
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test))
display_history(history)

# Evaluate.
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#%%-------------------------------------------------------------------
# CNN + LSTM.

# Embedding.
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution.
kernel_size = 5
filters = 64
pool_size = 4

# LSTM.
lstm_output_size = 70

# Training.
batch_size = 30
num_epochs = 2

# NOTICE [note] >>
#	batch_size is highly sensitive.
#	Only 2 epochs are needed as the dataset is very small.

# Loading data.
x_train, y_train, x_test, y_test = load_dataset(max_features, maxlen)

# Build model.
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train.
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test))
display_history(history)

# Evaluate.
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
