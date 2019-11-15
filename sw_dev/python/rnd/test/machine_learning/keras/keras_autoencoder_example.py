#!/usr/bin/env python
# coding: UTF-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, ELU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape, UpSampling2D, MaxPooling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from PIL import Image

# REF [site] >> https://keras.io/examples/mnist_denoising_autoencoder/
def denoising_autoencoder_cnn_mnist_example():
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

# Reparameterization trick.
# Instead of sampling from Q(z|X), sample epsilon = N(0, I),
# z = z_mean + sqrt(var) * epsilon.
def sampling(args):
	"""Reparameterization trick by sampling from an isotropic unit Gaussian.
	# Arguments
		args (tensor): mean and log of variance of Q(z|X).
	# Returns
		z (tensor): sampled latent vector.
	"""

	z_mean, z_log_var = args
	batch = K.shape(z_mean)[0]
	dim = K.int_shape(z_mean)[1]
	# By default, random_normal has mean = 0 and std = 1.0.
	epsilon = K.random_normal(shape=(batch, dim))
	return z_mean + K.exp(0.5 * z_log_var) * epsilon

def plot_results(models, data, batch_size=128, model_name='vae_mnist'):
	"""Plots labels and MNIST digits as a function of the 2D latent vector.
	# Arguments
		models (tuple): encoder and decoder models.
		data (tuple): test data and label.
		batch_size (int): prediction batch size.
		model_name (string): which model is using this function.
	"""

	encoder, decoder = models
	x_test, y_test = data
	os.makedirs(model_name, exist_ok=True)

	filename = os.path.join(model_name, 'vae_mean.png')
	# Display a 2D plot of the digit classes in the latent space.
	z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
	plt.figure(figsize=(12, 10))
	plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
	plt.colorbar()
	plt.xlabel('z[0]')
	plt.ylabel('z[1]')
	plt.savefig(filename)
	plt.show()

	filename = os.path.join(model_name, 'digits_over_latent.png')
	# Display a 30x30 2D manifold of digits.
	n = 30
	digit_size = 28
	figure = np.zeros((digit_size * n, digit_size * n))
	# Linearly spaced coordinates corresponding to the 2D plot of digit classes in the latent space.
	grid_x = np.linspace(-4, 4, n)
	grid_y = np.linspace(-4, 4, n)[::-1]

	for i, yi in enumerate(grid_y):
		for j, xi in enumerate(grid_x):
			z_sample = np.array([[xi, yi]])
			x_decoded = decoder.predict(z_sample)
			digit = x_decoded[0].reshape(digit_size, digit_size)
			figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

	plt.figure(figsize=(10, 10))
	start_range = digit_size // 2
	end_range = (n - 1) * digit_size + start_range + 1
	#end_range = n * digit_size + start_range + 1
	pixel_range = np.arange(start_range, end_range, digit_size)
	sample_range_x = np.round(grid_x, 1)
	sample_range_y = np.round(grid_y, 1)
	plt.xticks(pixel_range, sample_range_x)
	plt.yticks(pixel_range, sample_range_y)
	plt.xlabel('z[0]')
	plt.ylabel('z[1]')
	plt.imshow(figure, cmap='Greys_r')
	plt.savefig(filename)
	plt.show()

# REF [site] >> https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# REF [paper] >> "Auto-Encoding Variational Bayes", arXiv 2014.
def variational_autoencoder_mlp_mnist_example():
	is_mse_used = True  # Use MSE loss or binary cross entropy loss.
	is_trained_model_used = False
	weight_filepath = 'vae_mlp_mnist.h5'  # h5 model trained weights.

	#--------------------
	# MNIST dataset.
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	image_height, image_width = x_train.shape[1:3]
	image_size = image_height * image_width
	x_train = np.reshape(x_train, [-1, image_size])
	x_test = np.reshape(x_test, [-1, image_size])
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	#--------------------
	# Network parameters.
	input_shape = (image_size,)
	intermediate_dim = 512
	batch_size = 128
	latent_dim = 2
	epochs = 50

	#--------------------
	# VAE model = encoder + decoder.

	#--------------------
	# Build encoder model.
	inputs = Input(shape=input_shape, name='encoder_input')
	x = Dense(intermediate_dim, activation='relu')(inputs)
	z_mean = Dense(latent_dim, name='z_mean')(x)
	z_log_var = Dense(latent_dim, name='z_log_var')(x)

	# Use reparameterization trick to push the sampling out as input.
	# Note that 'output_shape' isn't necessary with the TensorFlow backend.
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# Instantiate encoder model.
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	encoder.summary()

	plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

	#--------------------
	# Build decoder model.
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	x = Dense(intermediate_dim, activation='relu')(latent_inputs)
	outputs = Dense(image_size, activation='sigmoid')(x)

	# Instantiate decoder model.
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()

	plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

	#--------------------
	# Instantiate VAE model.
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae_mlp')

	#--------------------
	# VAE loss = mse_loss or xent_loss + kl_loss.
	if is_mse_used:
		reconstruction_loss = mse(inputs, outputs)
	else:
		reconstruction_loss = binary_crossentropy(inputs, outputs)
	reconstruction_loss *= image_size

	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5

	vae_loss = K.mean(reconstruction_loss + kl_loss)

	vae.add_loss(vae_loss)
	vae.compile(optimizer='adam')
	vae.summary()

	plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

	#--------------------
	if is_trained_model_used:
		vae.load_weights(weight_filepath)
	else:
		# Train the autoencoder.
		vae.fit(x_train,
				epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_test, None))
		vae.save_weights(weight_filepath)

	#--------------------
	models = (encoder, decoder)
	data = (x_test, y_test)

	plot_results(models, data, batch_size=batch_size, model_name='vae_mlp')

# REF [site] >> https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py
def variational_autoencoder_cnn_mnist_example():
	is_mse_used = True  # Use MSE loss or binary cross entropy loss.
	is_trained_model_used = False
	weight_filepath = 'vae_cnn_mnist.h5'  # h5 model trained weights.

	#--------------------
	# MNIST dataset.
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	image_height, image_width = x_train.shape[1:3]
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	#--------------------
	# Network parameters.
	input_shape = (image_height, image_width, 1)
	batch_size = 128
	kernel_size = 3
	layer_filters = [32, 64]
	latent_dim = 2
	epochs = 30

	#--------------------
	# VAE model = encoder + decoder.

	#--------------------
	# Build encoder model.
	inputs = Input(shape=input_shape, name='encoder_input')
	x = inputs
	for filters in layer_filters:
		x = Conv2D(filters=filters,
				   kernel_size=kernel_size,
				   activation='relu',
				   strides=2,
				   padding='same')(x)

	# Shape info needed to build decoder model.
	shape = K.int_shape(x)

	# Generate latent vector Q(z|X).
	x = Flatten()(x)
	x = Dense(16, activation='relu')(x)
	z_mean = Dense(latent_dim, name='z_mean')(x)
	z_log_var = Dense(latent_dim, name='z_log_var')(x)

	# Use reparameterization trick to push the sampling out as input.
	# Note that 'output_shape' isn't necessary with the TensorFlow backend.
	z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

	# Instantiate encoder model.
	encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
	encoder.summary()

	plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

	#--------------------
	# Build decoder model.
	latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
	x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
	x = Reshape((shape[1], shape[2], shape[3]))(x)

	for filters in layer_filters[::-1]:
		x = Conv2DTranspose(filters=filters,
							kernel_size=kernel_size,
							activation='relu',
							strides=2,
							padding='same')(x)

	outputs = Conv2DTranspose(filters=1,
							  kernel_size=kernel_size,
							  activation='sigmoid',
							  padding='same',
							  name='decoder_output')(x)

	# Instantiate decoder model.
	decoder = Model(latent_inputs, outputs, name='decoder')
	decoder.summary()

	plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

	#--------------------
	# Instantiate VAE model.
	outputs = decoder(encoder(inputs)[2])
	vae = Model(inputs, outputs, name='vae')

	#--------------------
	# VAE loss = mse_loss or xent_loss + kl_loss
	if is_mse_used:
		reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
	else:
		reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
	reconstruction_loss *= image_height * image_width

	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss, axis=-1)
	kl_loss *= -0.5

	vae_loss = K.mean(reconstruction_loss + kl_loss)

	vae.add_loss(vae_loss)
	vae.compile(optimizer='rmsprop')
	vae.summary()

	plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

	#--------------------
	if is_trained_model_used:
		vae.load_weights(weight_filepath)
	else:
		# Train the autoencoder.
		vae.fit(x_train,
				epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_test, None))
		vae.save_weights(weight_filepath)

	#--------------------
	models = (encoder, decoder)
	data = (x_test, y_test)

	plot_results(models, data, batch_size=batch_size, model_name='vae_cnn')

def convresblock(x, nfeats=8, ksize=3, nskipped=2, elu=True):
	"""The proposed residual block from [4].
	Running with elu=True will use ELU nonlinearity and running with
	elu=False will use BatchNorm + RELU nonlinearity.  While ELU's are fast
	due to the fact they do not suffer from BatchNorm overhead, they may
	overfit because they do not offer the stochastic element of the batch
	formation process of BatchNorm, which acts as a good regularizer.
	# Arguments
		x: 4D tensor, the tensor to feed through the block
		nfeats: Integer, number of feature maps for conv layers.
		ksize: Integer, width and height of conv kernels in first convolution.
		nskipped: Integer, number of conv layers for the residual function.
		elu: Boolean, whether to use ELU or BN+RELU.
	# Input shape
		4D tensor with shape:
		`(batch, channels, rows, cols)`
	# Output shape
		4D tensor with shape:
		`(batch, filters, rows, cols)`
	"""
	y0 = Conv2D(nfeats, ksize, padding='same')(x)
	y = y0
	for i in range(nskipped):
		if elu:
			y = ELU()(y)
		else:
			y = BatchNormalization(axis=1)(y)
			y = Activation('relu')(y)
		y = Conv2D(nfeats, 1, padding='same')(y)
	return tf.keras.layers.add([y0, y])

def getwhere(x):
	"""Calculate the 'where' mask that contains switches indicating which
	index contained the max value when MaxPool2D was applied.  Using the
	gradient of the sum is a nice trick to keep everything high level.
	"""
	y_prepool, y_postpool = x
	return K.gradients(K.sum(y_postpool), y_prepool)

# REF [site] >> https://github.com/keras-team/keras/blob/master/examples/mnist_swwae.py
# REF [paper] >> "Stacked What-Where Auto-encoders", ICLRW 2016.
def stacked_what_where_autoencoder_cnn_mnist_example():
	#--------------------
	# This example assume 'channels_first' data format.
	K.set_image_data_format('channels_first')

	# Input image dimensions.
	img_rows, img_cols = 28, 28

	# The data, split between train and test sets.
	(x_train, _), (x_test, _) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	#--------------------
	# The size of the kernel used for the MaxPooling2D.
	pool_size = 2
	# The total number of feature maps at each layer.
	nfeats = [8, 16, 32, 64, 128]
	# The sizes of the pooling kernel at each layer.
	pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
	# The convolution kernel size.
	ksize = 3
	# Number of epochs to train for.
	epochs = 5
	# Batch size during training.
	batch_size = 128

	if pool_size == 2:
		# If using a 5 layer net of pool_size = 2.
		x_train = np.pad(x_train, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
		x_test = np.pad(x_test, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
		nlayers = 5
	elif pool_size == 3:
		# If using a 3 layer net of pool_size = 3.
		x_train = x_train[:,:,:-1,:-1]
		x_test = x_test[:,:,:-1,:-1]
		nlayers = 3
	else:
		import sys
		sys.exit('Script supports pool_size of 2 and 3.')

	# Shape of input to train on (note that model is fully convolutional however).
	input_shape = x_train.shape[1:]
	# The final list of the size of axis=1 for all layers, including input.
	nfeats_all = [input_shape[0]] + nfeats

	#--------------------
	# Build the encoder, all the while keeping track of the 'where' masks.
	img_input = Input(shape=input_shape)

	# We push the 'where' masks to the following list.
	wheres = [None] * nlayers
	y = img_input
	for i in range(nlayers):
		y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
		y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
		wheres[i] = tf.keras.layers.Lambda(getwhere, output_shape=lambda x: x[0])([y_prepool, y])[0]

	#--------------------
	# Build the decoder, and use the stored 'where' masks to place the features.
	for i in range(nlayers):
		ind = nlayers - 1 - i
		y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
		y = tf.keras.layers.multiply([y, wheres[ind]])
		y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

	# Use hard_simgoid to clip range of reconstruction.
	y = Activation('hard_sigmoid')(y)

	# Define the model.
	model = Model(img_input, y)

	#--------------------
	# Mean square error loss.
	model.compile('adam', 'mse')

	# Fit the model.
	model.fit(x_train, x_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  validation_data=(x_test, x_test))

	#--------------------
	# Plot.
	x_recon = model.predict(x_test[:25])
	x_plot = np.concatenate((x_test[:25], x_recon), axis=1)
	x_plot = x_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
	x_plot = np.vstack([np.hstack(x) for x in x_plot])
	plt.figure()
	plt.axis('off')
	plt.title('Test Samples: Originals/Reconstructions')
	plt.imshow(x_plot, interpolation='none', cmap='gray')
	plt.savefig('./reconstructions.png')

def main():
	denoising_autoencoder_cnn_mnist_example()

	#variational_autoencoder_mlp_mnist_example()
	#variational_autoencoder_cnn_mnist_example()

	#stacked_what_where_autoencoder_cnn_mnist_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
