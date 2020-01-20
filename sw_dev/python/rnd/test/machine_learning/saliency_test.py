#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/PAIR-code/saliency

import os, time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import saliency
from matplotlib import pylab as plt
import PIL.Image

# Boilerplate methods.
def ShowImage(im, title='', ax=None):
	if ax is None:
		plt.figure()
	plt.axis('off')
	im = ((im + 1) * 127.5).astype(np.uint8)
	plt.imshow(im)
	plt.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
	if ax is None:
		plt.figure()
	plt.axis('off')

	plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=1)
	plt.title(title)

def ShowHeatMap(im, title, ax=None):
	if ax is None:
		plt.figure()
	plt.axis('off')
	plt.imshow(im, cmap=plt.cm.inferno)
	plt.title(title)

def ShowDivergingImage(grad, title='', percentile=99, ax=None):  
	if ax is None:
		fig, ax = plt.subplots()
	else:
		fig = ax.figure

	plt.axis('off')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(grad, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
	fig.colorbar(im, cax=cax, orientation='vertical')
	plt.title(title)

def LoadImage(file_path):
	im = PIL.Image.open(file_path)
	im = np.asarray(im)
	return im / 127.5 - 1.0

# REF [site] >> https://github.com/PAIR-code/saliency/blob/master/Examples.ipynb
def simple_example():
	#--------------------
	mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

	#--------------------
	# Define a model.

	num_classes = 10
	input_shape = (None, 28, 28, 1)  # 784 = 28 * 28.
	output_shape = (None, num_classes)
	input_ph = tf.placeholder(tf.float32, shape=input_shape, name='input_ph')
	output_ph = tf.placeholder(tf.float32, shape=output_shape, name='output_ph')

	with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
		conv1 = tf.layers.conv2d(input_ph, 32, 5, activation=tf.nn.relu, name='conv')
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2, name='maxpool')

	with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
		conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu, name='conv')
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2, name='maxpool')

	with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		fc1 = tf.layers.flatten(conv2, name='flatten')
		fc1 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name='dense')

	with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
		model_output = tf.layers.dense(fc1, num_classes, activation=tf.nn.softmax, name='dense')

	#--------------------
	# Train.

	loss = tf.reduce_mean(-tf.reduce_sum(output_ph * tf.log(model_output), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	print('Start training...')
	start_time = time.time()
	for _ in range(2000):
		batch_xs, batch_ys = mnist.train.next_batch(512)
		batch_xs = np.reshape(batch_xs, (-1,) + input_shape[1:])
		sess.run(train_step, feed_dict={input_ph: batch_xs, output_ph: batch_ys})
	print('End training: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Evaluate.

	correct_prediction = tf.equal(tf.argmax(model_output, 1), tf.argmax(output_ph, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print('Start testing...')
	acc = sess.run(accuracy, feed_dict={input_ph: np.reshape(mnist.test.images, (-1,) + input_shape[1:]), output_ph: mnist.test.labels})
	print('Test accuracy = {}.'.format(acc))
	print('End testing: {} secs.'.format(time.time() - start_time))

	if acc < 0.95:
		print('Failed to train...')
		return

	#--------------------
	# Visualize.

	images = np.reshape(mnist.test.images, (-1,) + input_shape[1:])
	img = images[0]
	minval, maxval = np.min(img), np.max(img)
	img_scaled = np.squeeze((img - minval) / (maxval - minval), axis=-1)

	# Construct the scalar neuron tensor.
	logits = model_output
	neuron_selector = tf.placeholder(tf.int32)
	y = logits[0][neuron_selector]

	# Construct tensor for predictions.
	prediction = tf.argmax(logits, 1)

	# Make a prediction. 
	prediction_class = sess.run(prediction, feed_dict={input_ph: [img]})[0]

	print('Start visualizing saliency...')
	start_time = time.time()
	saliency_obj = saliency.Occlusion(sess.graph, sess, y, input_ph)

	# NOTE [info] >> An error exists in GetMask() of ${Saliency_HOME}/saliency/occlusion.py.
	#	<before>
	#		occlusion_window = np.array([size, size, x_value.shape[2]])
	#		occlusion_window.fill(value)
	#	<after>
	#		occlusion_window = np.full([size, size, x_value.shape[2]], value)
	mask_3d = saliency_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})

	# Compute a 2D tensor for visualization.
	mask_gray = saliency.VisualizeImageGrayscale(mask_3d)
	mask_div = saliency.VisualizeImageDiverging(mask_3d)

	fig = plt.figure()
	ax = plt.subplot(1, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(1, 3, 2)
	ax.imshow(mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Grayscale')
	ax = plt.subplot(1, 3, 3)
	ax.imshow(mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Diverging')
	fig.suptitle('Occlusion', fontsize=16)
	fig.tight_layout()
	plt.show()

	#--------------------
	conv_layer = sess.graph.get_tensor_by_name('conv2/conv/BiasAdd:0')
	saliency_obj = saliency.GradCam(sess.graph, sess, y, input_ph, conv_layer)

	mask_3d = saliency_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})

	# Compute a 2D tensor for visualization.
	mask_gray = saliency.VisualizeImageGrayscale(mask_3d)
	mask_div = saliency.VisualizeImageDiverging(mask_3d)

	fig = plt.figure()
	ax = plt.subplot(1, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(1, 3, 2)
	ax.imshow(mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Grayscale')
	ax = plt.subplot(1, 3, 3)
	ax.imshow(mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Diverging')
	fig.suptitle('Grad-CAM', fontsize=16)
	fig.tight_layout()
	plt.show()

	#--------------------
	saliency_obj = saliency.GradientSaliency(sess.graph, sess, y, input_ph)

	vanilla_mask_3d = saliency_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})
	smoothgrad_mask_3d = saliency_obj.GetSmoothedMask(img, feed_dict={neuron_selector: prediction_class})

	# Compute a 2D tensor for visualization.
	vanilla_mask_gray = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
	smoothgrad_mask_gray = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
	vanilla_mask_div = saliency.VisualizeImageDiverging(vanilla_mask_3d)
	smoothgrad_mask_div = saliency.VisualizeImageDiverging(smoothgrad_mask_3d)

	fig = plt.figure()
	ax = plt.subplot(2, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(2, 3, 2)
	ax.imshow(vanilla_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Grayscale')
	ax = plt.subplot(2, 3, 3)
	ax.imshow(smoothgrad_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Grayscale')
	ax = plt.subplot(2, 3, 4)
	ax.imshow(vanilla_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Diverging')
	ax = plt.subplot(2, 3, 5)
	ax.imshow(smoothgrad_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Diverging')
	fig.suptitle('Gradient Saliency', fontsize=16)
	fig.tight_layout()
	plt.show()

	#--------------------
	saliency_obj = saliency.GuidedBackprop(sess.graph, sess, y, input_ph)

	vanilla_mask_3d = saliency_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})
	smoothgrad_mask_3d = saliency_obj.GetSmoothedMask(img, feed_dict={neuron_selector: prediction_class})

	# Compute a 2D tensor for visualization.
	vanilla_mask_gray = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
	smoothgrad_mask_gray = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
	vanilla_mask_div = saliency.VisualizeImageDiverging(vanilla_mask_3d)
	smoothgrad_mask_div = saliency.VisualizeImageDiverging(smoothgrad_mask_3d)

	fig = plt.figure()
	ax = plt.subplot(2, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(2, 3, 2)
	ax.imshow(vanilla_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Grayscale')
	ax = plt.subplot(2, 3, 3)
	ax.imshow(smoothgrad_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Grayscale')
	ax = plt.subplot(2, 3, 4)
	ax.imshow(vanilla_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Diverging')
	ax = plt.subplot(2, 3, 5)
	ax.imshow(smoothgrad_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Diverging')
	fig.suptitle('Guided Backprop', fontsize=16)
	fig.tight_layout()
	plt.show()

	#--------------------
	saliency_obj = saliency.IntegratedGradients(sess.graph, sess, y, input_ph)

	vanilla_mask_3d = saliency_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})
	smoothgrad_mask_3d = saliency_obj.GetSmoothedMask(img, feed_dict={neuron_selector: prediction_class})

	# Compute a 2D tensor for visualization.
	vanilla_mask_gray = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
	smoothgrad_mask_gray = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
	vanilla_mask_div = saliency.VisualizeImageDiverging(vanilla_mask_3d)
	smoothgrad_mask_div = saliency.VisualizeImageDiverging(smoothgrad_mask_3d)

	fig = plt.figure()
	ax = plt.subplot(2, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(2, 3, 2)
	ax.imshow(vanilla_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Grayscale')
	ax = plt.subplot(2, 3, 3)
	ax.imshow(smoothgrad_mask_gray, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Grayscale')
	ax = plt.subplot(2, 3, 4)
	ax.imshow(vanilla_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Vanilla Diverging')
	ax = plt.subplot(2, 3, 5)
	ax.imshow(smoothgrad_mask_div, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('SmoothGrad Diverging')
	fig.suptitle('Integrated Gradients', fontsize=16)
	fig.tight_layout()
	plt.show()

	#--------------------
	xrai_obj = saliency.XRAI(sess.graph, sess, y, input_ph)

	if True:
		xrai_attributions = xrai_obj.GetMask(img, feed_dict={neuron_selector: prediction_class})
	else:
		# Create XRAIParameters and set the algorithm to fast mode which will produce an approximate result.
		xrai_params = saliency.XRAIParameters()
		xrai_params.algorithm = 'fast'
		xrai_attributions_fast = xrai_obj.GetMask(img, feed_dict={neuron_selector: prediction_class}, extra_parameters=xrai_params)

	# Show most salient 30% of the image.
	mask = xrai_attributions > np.percentile(xrai_attributions, 70)
	img_masked = img_scaled.copy()
	img_masked[~mask] = 0

	fig = plt.figure()
	ax = plt.subplot(1, 3, 1)
	ax.imshow(img_scaled, cmap=plt.cm.gray, vmin=0, vmax=1)
	ax.axis('off')
	ax.set_title('Input')
	ax = plt.subplot(1, 3, 2)
	ax.imshow(xrai_attributions, cmap=plt.cm.inferno)
	ax.axis('off')
	ax.set_title('XRAI Attributions')
	ax = plt.subplot(1, 3, 3)
	ax.imshow(img_masked, cmap=plt.cm.gray)
	ax.axis('off')
	ax.set_title('Masked Input')
	fig.suptitle('XRAI', fontsize=16)
	fig.tight_layout()
	plt.show()
	print('End visualizing saliency: {} secs.'.format(time.time() - start_time))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
