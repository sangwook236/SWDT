#!/usr/bin/env python
# coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image

# REF [site] >> https://keras.io/examples/mnist_denoising_autoencoder/
def mnist_denoising_autoencoder_example():
	np.random.seed(1337)

	#--------------------
	# MNIST dataset.
	(x_train, _), (x_test, _) = mnist.load_data()

	image_height, image_width = x_train.shape[1:3]
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	# Generate corrupted MNIST images by adding noise with normal dist centered at 0.5 and std = 0.5.
	noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
	x_train_noisy = x_train + noise
	noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
	x_test_noisy = x_test + noise

	x_train_noisy = np.clip(x_train_noisy, 0., 1.)
	x_test_noisy = np.clip(x_test_noisy, 0., 1.)

	#--------------------
	# Network parameters.
	input_shape = (image_height, image_width, 1)
	batch_size = 128
	kernel_size = 3
	latent_dim = 16
	# Encoder/Decoder number of CNN layers and filters per layer.
	layer_filters = [32, 64]

	#--------------------
	# Build the Autoencoder Model.

	#--------------------
	# Build the Encoder Model.
	inputs = Input(shape=input_shape, name='encoder_input')
	x = inputs
	# Stack of Conv2D blocks.
	# Notes:
	# 1) Use Batch Normalization before ReLU on deep networks.
	# 2) Use MaxPooling2D as alternative to strides > 1.
	# - faster but not as good as strides > 1.
	for filters in layer_filters:
		x = Conv2D(filters=filters,
				   kernel_size=kernel_size,
				   strides=2,
				   activation='relu',
				   padding='same')(x)

	# Shape info needed to build Decoder Model.
	shape = K.int_shape(x)

	# Generate the latent vector.
	x = Flatten()(x)
	latent = Dense(latent_dim, name='latent_vector')(x)

	# Instantiate Encoder Model.
	encoder = Model(inputs, latent, name='encoder')
	encoder.summary()

	#--------------------
	# Build the Decoder Model.
	latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
	x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
	x = Reshape((shape[1], shape[2], shape[3]))(x)

	# Stack of Transposed Conv2D blocks.
	# Notes:
	# 1) Use Batch Normalization before ReLU on deep networks.
	# 2) Use UpSampling2D as alternative to strides > 1.
	# - faster but not as good as strides > 1.
	for filters in layer_filters[::-1]:
		x = Conv2DTranspose(filters=filters,
							kernel_size=kernel_size,
							strides=2,
							activation='relu',
							padding='same')(x)

	x = Conv2DTranspose(filters=1,
						kernel_size=kernel_size,
						padding='same')(x)

	outputs = Activation('sigmoid', name='decoder_output')(x)

	# Instantiate Decoder Model.
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()

	#--------------------
	# Autoencoder = Encoder + Decoder.
	# Instantiate Autoencoder Model.
	autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
	autoencoder.summary()

	#--------------------
	autoencoder.compile(loss='mse', optimizer='adam')

	#--------------------
	# Train the autoencoder.
	autoencoder.fit(x_train_noisy,
					x_train,
					validation_data=(x_test_noisy, x_test),
					epochs=30,
					batch_size=batch_size)

	#--------------------
	# Predict the Autoencoder output from corrupted test images.
	x_decoded = autoencoder.predict(x_test_noisy)

	#--------------------
	# Display the 1st 8 corrupted and denoised images.
	rows, cols = 10, 30
	num = rows * cols
	imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
	imgs = imgs.reshape((rows * 3, cols, image_height, image_width))
	imgs = np.vstack(np.split(imgs, rows, axis=1))
	imgs = imgs.reshape((rows * 3, -1, image_height, image_width))
	imgs = np.vstack([np.hstack(i) for i in imgs])
	imgs = (imgs * 255).astype(np.uint8)
	plt.figure()
	plt.axis('off')
	plt.title('Original images: top rows, '
			  'Corrupted Input: middle rows, '
			  'Denoised Input:  third rows')
	plt.imshow(imgs, interpolation='none', cmap='gray')
	Image.fromarray(imgs).save('corrupted_and_denoised.png')
	plt.show()

def main():
	mnist_denoising_autoencoder_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
