#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# REF [site] >> https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
def visualize_filters_in_CNN():
	# Load the model.
	model = tf.keras.applications.vgg16.VGG16()
	# Summarize the model.
	model.summary()

	# Summarize filter shapes.
	for layer in model.layers:
		# Check for convolutional layer.
		if 'conv' not in layer.name:
			continue

		# Get filter weights.
		filters, biases = layer.get_weights()
		print(layer.name, filters.shape)

	#--------------------
	# Retrieve weights from the second hidden layer.
	filters, biases = model.layers[1].get_weights()
	# Normalize filter values to 0-1 so we can visualize them.
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min)

	# Plot first few filters.
	n_filters, ix = 6, 1
	for i in range(n_filters):
		# Get the filter.
		f = filters[:, :, :, i]
		# Plot each channel separately.
		for j in range(3):
			# Specify subplot and turn of axis.
			ax = plt.subplot(n_filters, 3, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# Plot filter channel in grayscale.
			plt.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# Show the figure.
	plt.show()

# REF [site] >> https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
def visualize_feature_maps_in_CNN():
	# Load the image with the required shape.
	img = tf.keras.preprocessing.image.load_img('./bird.jpg', target_size=(224, 224))
	# Convert the image to an array.
	img = tf.keras.preprocessing.image.img_to_array(img)
	# Expand dimensions so that it represents a single 'sample'.
	img = np.expand_dims(img, axis=0)

	# Prepare the image (e.g. scale pixel values for the vgg).
	img = tf.keras.applications.vgg16.preprocess_input(img)

	#--------------------
	# Load the model.
	model = tf.keras.applications.vgg16.VGG16()

	# Redefine model to output right after the first hidden layer.
	model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)
	model.summary()

	# Get feature map for first hidden layer.
	feature_maps = model.predict(img)

	# Plot all 64 maps in an 8x8 squares.
	square = 8
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# Specify subplot and turn of axis.
			ax = plt.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# Plot filter channel in grayscale.
			plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
			ix += 1
	# Show the figure.
	plt.show()

	#--------------------
	# Load the model.
	model = tf.keras.applications.vgg16.VGG16()

	# Redefine model to output right after hidden layers.
	ixs = [2, 5, 9, 13, 17]
	outputs = [model.layers[i].output for i in ixs]
	model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)

	# Get feature map for the hidden layers.
	feature_maps = model.predict(img)

	# Plot the output from each block.
	square = 8
	for fmap in feature_maps:
		# Plot all 64 maps in an 8x8 squares.
		ix = 1
		for _ in range(square):
			for _ in range(square):
				# Specify subplot and turn of axis.
				ax = plt.subplot(square, square, ix)
				ax.set_xticks([])
				ax.set_yticks([])
				# Plot filter channel in grayscale.
				plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
				ix += 1
		# Show the figure.
		plt.show()

def main():
	#visualize_filters_in_CNN()
	visualize_feature_maps_in_CNN()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
