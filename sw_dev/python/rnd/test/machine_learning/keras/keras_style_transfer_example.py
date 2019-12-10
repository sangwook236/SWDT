#!/usr/bin/env python
# coding: UTF-8

from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications import vgg19
from tensorflow.keras import backend as K
from scipy.optimize import fmin_l_bfgs_b

# Util function to open, resize and format pictures into appropriate tensors.
def preprocess_image(image_path, img_nrows, img_ncols):
	img = load_img(image_path, target_size=(img_nrows, img_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = vgg19.preprocess_input(img, K.image_data_format())
	return img

# Util function to convert a tensor into a valid image.
def deprocess_image(x, img_nrows, img_ncols):
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, img_nrows, img_ncols))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((img_nrows, img_ncols, 3))
	# Remove zero-center by mean pixel.
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	# BGR -> RGB.
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype('uint8')
	return x

# The gram matrix of an image tensor (feature-wise outer product).
def gram_matrix(x):
	assert K.ndim(x) == 3
	if K.image_data_format() == 'channels_first':
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

# The "style loss" is designed to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image
def style_loss(style, combination, img_nrows, img_ncols):
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = img_nrows * img_ncols
	return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# An auxiliary loss function designed to maintain the "content" of the base image in the generated image.
def content_loss(base, combination):
	return K.sum(K.square(combination - base))

# The 3rd loss function, total variation loss, designed to keep the generated image locally coherent
def total_variation_loss(x, img_nrows, img_ncols):
	assert K.ndim(x) == 4
	if K.image_data_format() == 'channels_first':
		a = K.square( x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
		b = K.square( x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
	else:
		a = K.square( x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
		b = K.square( x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))

def eval_loss_and_grads(x, img_nrows, img_ncols, f_outputs):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# This Evaluator class makes it possible to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
	def __init__(self, img_nrows, img_ncols, f_outputs):
		self.loss_value = None
		self.grads_values = None
		self.img_nrows = img_nrows
		self.img_ncols = img_ncols
		self.f_outputs = f_outputs

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x, self.img_nrows, self.img_ncols, self.f_outputs)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values

# REF [site] >>
#	https://keras.io/examples/neural_style_transfer/
#	https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
# REF [paper] >> "A Neural Algorithm of Artistic Style", arXiv 2015.
def neural_style_transfer_example():
	base_image_filepath = './Machu_Picchu.jpg'  # Path to the image to transform.
	style_reference_image_filepath = './Starry_Night.jpg'  # Path to the style reference image.
	result_prefix = './neural_style_transfer_results'  # Prefix for the saved results.
	iterations = 10  # Number of iterations to run.

	# These are the weights of the different loss components.
	total_variation_weight = 1.0  # Total Variation weight.
	style_weight = 1.0  # Style weight.
	content_weight = 0.025  # Content weight.

	# Dimensions of the generated picture.
	width, height = load_img(base_image_filepath).size
	img_nrows = 400
	img_ncols = int(width * img_nrows / height)

	# Get tensor representations of our images.
	base_image = K.variable(preprocess_image(base_image_filepath, img_nrows, img_ncols))
	style_reference_image = K.variable(preprocess_image(style_reference_image_filepath, img_nrows, img_ncols))

	# This will contain our generated image.
	if K.image_data_format() == 'channels_first':
		combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
	else:
		combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

	# Combine the 3 images into a single Keras tensor.
	input_tensor = K.concatenate([base_image,
								  style_reference_image,
								  combination_image], axis=0)

	# Build the VGG19 network with our 3 images as input.
	# The model will be loaded with pre-trained ImageNet weights.
	model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False, backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils)
	print('Model loaded.')

	# Get the symbolic outputs of each "key" layer (we gave them unique names).
	outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

	# Compute the neural style loss.
	# Combine these loss functions into a single scalar.
	loss = K.variable(0.0)
	layer_features = outputs_dict['block5_conv2']
	base_image_features = layer_features[0, :, :, :]
	combination_features = layer_features[2, :, :, :]
	loss = loss + content_weight * content_loss(base_image_features, combination_features)

	feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
	for layer_name in feature_layers:
		layer_features = outputs_dict[layer_name]
		style_reference_features = layer_features[1, :, :, :]
		combination_features = layer_features[2, :, :, :]
		sl = style_loss(style_reference_features, combination_features, img_nrows, img_ncols)
		loss = loss + (style_weight / len(feature_layers)) * sl
	loss = loss + total_variation_weight * total_variation_loss(combination_image, img_nrows, img_ncols)

	# Get the gradients of the generated image wrt the loss.
	grads = K.gradients(loss, combination_image)

	outputs = [loss]
	if isinstance(grads, (list, tuple)):
		outputs += grads
	else:
		outputs.append(grads)

	f_outputs = K.function([combination_image], outputs)

	evaluator = Evaluator(img_nrows, img_ncols, f_outputs)

	# Run scipy-based optimization (L-BFGS) over the pixels of the generated image so as to minimize the neural style loss.
	x = preprocess_image(base_image_filepath, img_nrows, img_ncols)

	for i in range(iterations):
		print('Start of iteration', i)
		start_time = time.time()
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
		print('Current loss value:', min_val)
		# Save current generated image.
		img = deprocess_image(x.copy(), img_nrows, img_ncols)
		fname = result_prefix + '_at_iteration_%d.png' % i
		save_img(fname, img)
		end_time = time.time()
		print('Image saved as', fname)
		print('Iteration %d completed in %ds' % (i, end_time - start_time))

def main():
	neural_style_transfer_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
