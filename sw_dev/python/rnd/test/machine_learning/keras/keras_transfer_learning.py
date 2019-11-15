#!/usr/bin/env python
# coding: UTF-8

from __future__ import print_function

import datetime
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

def train_model(model, train, test, num_classes, input_shape, batch_size, epochs):
	x_train = train[0].reshape((train[0].shape[0],) + input_shape)
	x_test = test[0].reshape((test[0].shape[0],) + input_shape)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = tf.keras.utils.to_categorical(train[1], num_classes)
	y_test = tf.keras.utils.to_categorical(test[1], num_classes)

	model.compile(loss='categorical_crossentropy',
				  optimizer='adadelta',
				  metrics=['accuracy'])

	t = datetime.datetime.now()
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1,
			  validation_data=(x_test, y_test))
	print('Training time: %s' % (datetime.datetime.now() - t))
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

# REF [site] >> https://github.com/keras-team/keras/blob/master/examples/mnist_transfer_cnn.py
def simple_transfer_learning_cnn_mnist_example():
	batch_size = 128
	num_classes = 5
	epochs = 5

	# Input image dimensions.
	img_rows, img_cols = 28, 28
	# Number of convolutional filters to use.
	filters = 32
	# Size of pooling area for max pooling.
	pool_size = 2
	# Convolution kernel size.
	kernel_size = 3

	if K.image_data_format() == 'channels_first':
		input_shape = (1, img_rows, img_cols)
	else:
		input_shape = (img_rows, img_cols, 1)

	#--------------------
	# The data, split between train and test sets.
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Create two datasets one with digits below 5 and one with 5 and above.
	x_train_lt5 = x_train[y_train < 5]
	y_train_lt5 = y_train[y_train < 5]
	x_test_lt5 = x_test[y_test < 5]
	y_test_lt5 = y_test[y_test < 5]

	x_train_gte5 = x_train[y_train >= 5]
	y_train_gte5 = y_train[y_train >= 5] - 5
	x_test_gte5 = x_test[y_test >= 5]
	y_test_gte5 = y_test[y_test >= 5] - 5

	#--------------------
	# Define two groups of layers: feature (convolutions) and classification (dense).
	feature_layers = [
		Conv2D(filters, kernel_size, padding='valid', input_shape=input_shape),
		Activation('relu'),
		Conv2D(filters, kernel_size),
		Activation('relu'),
		MaxPooling2D(pool_size=pool_size),
		Dropout(0.25),
		Flatten(),
	]

	classification_layers = [
		Dense(128),
		Activation('relu'),
		Dropout(0.5),
		Dense(num_classes),
		Activation('softmax')
	]

	# Create complete model.
	model = Sequential(feature_layers + classification_layers)

	#--------------------
	# Train model for 5-digit classification [0..4].
	train_model(model,
				(x_train_lt5, y_train_lt5),
				(x_test_lt5, y_test_lt5),
				num_classes, input_shape, batch_size, epochs)

	# Freeze feature layers and rebuild model.
	for layer in feature_layers:
		layer.trainable = False

	#--------------------
	# Transfer: train dense layers for new classification task [5..9].
	train_model(model,
				(x_train_gte5, y_train_gte5),
				(x_test_gte5, y_test_gte5),
				num_classes, input_shape, batch_size, epochs)

def main():
	simple_transfer_learning_cnn_mnist_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
