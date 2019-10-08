# REF [site] >> https://bigsnarf.wordpress.com/2016/10/19/visualizing-convnet-layers-and-activations/
# REF [site] >> https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4
# REF [paper] >> "Visualizing and Understanding Convolutional Networks", ECCV 2014.

#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

#--------------------------------------------------------------------

import os
if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

mnist = input_data.read_data_sets(data_home_dir_path + '/pattern_recognition/mnist/0_original', one_hot=True)

#--------------------------------------------------------------------

def weight_variable(shape, name):
	#initial = tf.truncated_normal(shape, stddev=0.1)
	#return tf.Variable(initial, name=name)
	return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

def bias_variable(shape, name):
	#initial = tf.constant(0.1, shape=shape)
	#return tf.Variable(initial, name=name)
	return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.1))

#--------------------------------------------------------------------
# Creqte a model

#tf.reset_default_graph()

image_ph = tf.placeholder(tf.float32, [None, 784], name='input_tensor')
label_ph = tf.placeholder(tf.float32, [None, 10], name='output_tensor')
keep_prob_ph = tf.placeholder(tf.float32)

image_tensor = tf.reshape(image_ph, [-1, 28, 28, 1])

if True:
	# Uses TensorFlow.

	with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
		conv1_W = weight_variable([5, 5, 1, 5], name='weights')
		conv1_b = bias_variable([5], name='biases')
		conv1_preact = tf.nn.conv2d(input=image_tensor, filter=conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
		conv1_act = tf.nn.relu(conv1_preact)
		conv1_pool = tf.nn.max_pool(conv1_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

	with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
		conv2_W = weight_variable([5, 5, 5, 10], name='weights')
		conv2_b = bias_variable([10], name='biases')
		conv2_preact = tf.nn.conv2d(input=conv1_pool, filter=conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
		conv2_act = tf.nn.relu(conv2_preact)
		conv2_pool = tf.nn.max_pool(conv2_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

	with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
		conv3_W = weight_variable([5, 5, 10, 20], name='weights')
		conv3_b = bias_variable([20], name='biases')
		conv3_preact = tf.nn.conv2d(input=conv2_pool, filter=conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
		conv3_act = tf.nn.relu(conv3_preact)
		conv3_dropout = tf.nn.dropout(conv3_act, keep_prob_ph)

		conv3_flat = tf.reshape(conv3_dropout, [-1, 7 * 7 * 20])

	with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		fc1_W = weight_variable([7 * 7 * 20, 10], name='weights')
		fc1_b = bias_variable([10], name='biases')
		fc1_preact = tf.matmul(conv3_flat, fc1_W) + fc1_b
		fc1_act = tf.nn.softmax(fc1_preact)
else:
	# Uses TF-Slim.

	with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
		conv1_act = slim.conv2d(image_tensor, 5, [5, 5])
		conv1_pool = slim.max_pool2d(conv1_act, [2, 2])

	with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
		conv2_act = slim.conv2d(conv1_pool, 10, [5, 5])
		conv2_pool = slim.max_pool2d(conv2_act,[2, 2])

	with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
		conv3_act = slim.conv2d(conv2_pool, 20, [5, 5])
		conv3_dropout = slim.dropout(conv3_act, keep_prob_ph)

	with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		fc1_act = slim.fully_connected(slim.flatten(conv3_dropout), 10, activation_fn=tf.nn.softmax)

model_output_tensor = fc1_act

if True:
	global_variables = tf.global_variables()
	#print(global_variables)
	for var in global_variables:
		print(var)

#--------------------------------------------------------------------

cross_entropy = -tf.reduce_sum(label_ph * tf.log(model_output_tensor))
correct_prediction = tf.equal(tf.argmax(model_output_tensor, -1), tf.argmax(label_ph, -1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#--------------------------------------------------------------------
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

#--------------------------------------------------------------------

def plot_conv_filters(sess, filter_variable, num_columns=5, figsize=None):
	filters = filter_variable.eval(sess)  # Shape = (height, width, input_dim, output_dim).
	input_dim, output_dim = filters.shape[2], filters.shape[3]
	num_columns = num_columns if num_columns > 0 else 1
	num_rows = math.ceil(output_dim / num_columns) + 1
	for odim in range(output_dim):
		plt.figure(figsize=figsize)
		for idim in range(input_dim):
			plt.subplot(num_rows, num_columns, idim + 1)
			#plt.title('Filter {}'.format(idim))
			plt.imshow(filters[:,:,idim,odim], interpolation='nearest', cmap='gray')

#--------------------------------------------------------------------
# Visualize filters in a convolutional layer.

# NOTE [info] >> Need training.
with session.as_default() as sess:
	if True:
		# Uses TensorFlow.

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('weights')
			plot_conv_filters(sess, filters)
		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('weights')
			plot_conv_filters(sess, filters)
		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			filters = tf.get_variable('weights')
			plot_conv_filters(sess, filters)
		#with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		#	filters = tf.get_variable('weights')
		#	plot_conv_filters(sess, filters)
	else:
		# Uses TF-Slim.

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('Conv', reuse=tf.AUTO_REUSE):
				filters = tf.get_variable('weights')
				plot_conv_filters(sess, filters)
		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('Conv', reuse=tf.AUTO_REUSE):
				filters = tf.get_variable('weights')
				plot_conv_filters(sess, filters)
		with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE):
			with tf.variable_scope('Conv', reuse=tf.AUTO_REUSE):
				filters = tf.get_variable('weights')
				plot_conv_filters(sess, filters)
		#with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		#	with tf.variable_scope('fully_connected', reuse=tf.AUTO_REUSE):
		#		filters = tf.get_variable('weights')
		#		plot_conv_filters(sess, filters)
