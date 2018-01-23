# REF [site] >> https://bigsnarf.wordpress.com/2016/10/19/visualizing-convnet-layers-and-activations/
# REF [site] >> https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# REF [paper] >> "Visualizing and Understanding Convolutional Networks", ECCV 2014.

import numpy as np 
#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

#%%------------------------------------------------------------------

import os
if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

mnist = input_data.read_data_sets(data_home_dir_path + '/pattern_recognition/mnist/0_original', one_hot=True)

#%%------------------------------------------------------------------

def weight_variable(shape, name=None):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

#%%------------------------------------------------------------------
# Create a model.
	
#tf.reset_default_graph()

image_ph = tf.placeholder(tf.float32, [None, 784], name='input_tensor')
label_ph = tf.placeholder(tf.float32, [None, 10], name='output_tensor')
keep_prob_ph = tf.placeholder(tf.float32)

image_tensor = tf.reshape(image_ph, [-1, 28, 28, 1])

if True:
	# Uses TensorFlow.

	conv1_W = weight_variable([5, 5, 1, 5])
	conv1_b = bias_variable([5])
	conv1_preact = tf.nn.conv2d(input=image_tensor, filter=conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
	conv1_act = tf.nn.relu(conv1_preact)
	conv1_pool = tf.nn.max_pool(conv1_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

	conv2_W = weight_variable([5, 5, 5, 10])
	conv2_b = bias_variable([10])
	conv2_preact = tf.nn.conv2d(input=conv1_pool, filter=conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
	conv2_act = tf.nn.relu(conv2_preact)
	conv2_pool = tf.nn.max_pool(conv2_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

	conv3_W = weight_variable([5, 5, 10, 20])
	conv3_b = bias_variable([20])
	conv3_preact = tf.nn.conv2d(input=conv2_pool, filter=conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
	conv3_act = tf.nn.relu(conv3_preact)
	conv3_dropout = tf.nn.dropout(conv3_act, keep_prob_ph)

	conv3_flat = tf.reshape(conv3_dropout, [-1, 7 * 7 * 20])

	fc1_W = weight_variable([7 * 7 * 20, 10])
	fc1_b = bias_variable([10])
	fc1_preact = tf.matmul(conv3_flat, fc1_W) + fc1_b
	fc1_act = tf.nn.softmax(fc1_preact)
else:
	# Uses TF-Slim.

	conv1_act = slim.conv2d(image_tensor, 5, [5, 5])
	conv1_pool = slim.max_pool2d(conv1_act, [2, 2])

	conv2_act = slim.conv2d(conv1_pool, 10, [5, 5])
	conv2_pool = slim.max_pool2d(conv2_act,[2, 2])

	conv3_act = slim.conv2d(conv2_pool, 20, [5, 5])
	conv3_dropout = slim.dropout(conv3_act, keep_prob_ph)

	fc1_act = slim.fully_connected(slim.flatten(conv3_dropout), 10, activation_fn=tf.nn.softmax)

model_output_tensor = fc1_act

#%%------------------------------------------------------------------

cross_entropy = -tf.reduce_sum(label_ph * tf.log(model_output_tensor))
correct_prediction = tf.equal(tf.argmax(model_output_tensor, -1), tf.argmax(label_ph, -1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#%%------------------------------------------------------------------
# Train.

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

with session.as_default() as sess:
	batch_size = 50
	for step in range(1, 1001):
		batch = mnist.train.next_batch(batch_size)
		sess.run(train_step, feed_dict={image_ph: batch[0], label_ph: batch[1], keep_prob_ph: 0.5})
		if 0 == step % 100:
			train_acc = sess.run(accuracy, feed_dict={image_ph: batch[0], label_ph: batch[1], keep_prob_ph: 1.0})
			print('Step %d: training accuracy = %g' % (step, train_acc))

	test_acc = sess.run(accuracy, feed_dict={image_ph: mnist.test.images, label_ph: mnist.test.labels, keep_prob_ph: 1.0})
	print('Test accuracy = %g' % (test_acc))

#%%------------------------------------------------------------------

def plot_conv_activations(activations, num_columns=5, figsize=None):
	num_layers = activations.shape[3]
	plt.figure(figsize=figsize)
	num_columns = num_columns if num_columns > 0 else 1
	num_rows = math.ceil(num_layers / num_columns) + 1
	for i in range(num_layers):
		plt.subplot(num_rows, num_columns, i + 1)
		plt.title('Layer output {}'.format(i))
		plt.imshow(activations[0,:,:,i], interpolation='nearest', cmap='gray')

def compute_layer_activations(sess, layer_tensor, input_stimuli):
	return sess.run(layer_tensor, feed_dict={image_ph: np.reshape(input_stimuli, [1, 784], order='F'), keep_prob_ph: 1.0})  # Neurons -> numpy.array.

#%%------------------------------------------------------------------
# Visualize activations(layer ouputs) in a convolutional layer.

with session.as_default() as sess:
	imageToUse = mnist.test.images[0]
	plt.imshow(np.reshape(imageToUse, [28, 28]), interpolation='nearest', cmap='gray')

	# NOTE [info] >> Another way of obtaining a tf.Tensor object tensorflow_activation_visualization_2.py.
	#	A tf.Tensor object is retrieved using tensor's name & tf.Graph.get_tensor_by_name().

	#activations = compute_layer_activations(sess, conv1_preact, imageToUse)
	#plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv1_act, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv1_pool, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
	#activations = compute_layer_activations(sess, conv2_preact, imageToUse)
	#plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv2_act, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv2_pool, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
	#activations = compute_layer_activations(sess, conv3_preact, imageToUse)
	#plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv3_act, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
	activations = compute_layer_activations(sess, conv3_dropout, imageToUse)
	plot_conv_activations(activations, figsize=(40, 40))
