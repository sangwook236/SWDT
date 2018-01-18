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

def create_tf_model(input_tensor, keep_prob_tensor):
	with tf.variable_scope('my_tf_scope', reuse=tf.AUTO_REUSE):  # Applied variables and operations.
	#with tf.name_scope('my_tf_scope'):  # Applied variables and operations. (?)
		image_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])

		conv1_W = weight_variable([5, 5, 1, 32])
		conv1_b = bias_variable([32])
		conv1_preact = tf.nn.conv2d(input=image_tensor, filter=conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
		conv1_act = tf.nn.relu(conv1_preact)
		conv1_pool = tf.nn.max_pool(conv1_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

		conv2_W = weight_variable([5, 5, 32, 64])
		conv2_b = bias_variable([64])
		conv2_preact = tf.nn.conv2d(input=conv1_pool, filter=conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
		conv2_act = tf.nn.relu(conv2_preact)
		conv2_pool = tf.nn.max_pool(conv2_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

		conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])

		fc1_W = weight_variable([7 * 7 * 64, 1024])
		fc1_b = bias_variable([1024])
		fc1_preact = tf.matmul(conv2_flat, fc1_W) + fc1_b
		fc1_act = tf.nn.relu(fc1_preact)
		fc1_dropout = tf.nn.dropout(fc1_act, keep_prob_tensor)

		fc2_W = weight_variable([1024, 10])
		fc2_b = bias_variable([10])
		fc2_preact = tf.matmul(fc1_dropout, fc2_W) + fc2_b
		fc2_act = tf.nn.softmax(fc2_preact)

		return fc2_act

def create_tf_named_model(input_tensor, keep_prob_tensor):
	with tf.variable_scope('my_tf_named_scope', reuse=tf.AUTO_REUSE):  # Applied variables and operations.
	#with tf.name_scope('my_tf_named_scope'):  # Applied variables and operations. (?)
		image_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])

		with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE):
		#with tf.name_scope('conv1_act'):
			conv1_W = weight_variable([5, 5, 1, 32], name='weights')
			conv1_b = bias_variable([32], name='biases')
			conv1_preact = tf.nn.conv2d(input=image_tensor, filter=conv1_W, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
			conv1_act = tf.nn.relu(conv1_preact, name='activations')
			conv1_pool = tf.nn.max_pool(conv1_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pooling')

		with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE):
		#with tf.name_scope('conv2'):
			conv2_W = weight_variable([5, 5, 32, 64], name='weights')
			conv2_b = bias_variable([64], name='biases')
			conv2_preact = tf.nn.conv2d(input=conv1_pool, filter=conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
			conv2_act = tf.nn.relu(conv2_preact, name='activations')
			conv2_pool = tf.nn.max_pool(conv2_act, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME', name='pooling')

			conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64], name='flatten')

		with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
		#with tf.name_scope('fc1_act'):
			fc1_W = weight_variable([7 * 7 * 64, 1024], name='weights')
			fc1_b = bias_variable([1024], name='biases')
			fc1_preact = tf.matmul(conv2_flat, fc1_W) + fc1_b
			fc1_act = tf.nn.relu(fc1_preact, name='activations')
			fc1_dropout = tf.nn.dropout(fc1_act, keep_prob_tensor, name='dropout')

		with tf.variable_scope('fc2', reuse=tf.AUTO_REUSE):
		#with tf.name_scope('fc2_act'):
			fc2_W = weight_variable([1024, 10], name='weights')
			fc2_b = bias_variable([10], name='biases')
			fc2_preact = tf.matmul(fc1_dropout, fc2_W) + fc2_b
			fc2_act = tf.nn.softmax(fc2_preact, name='activations')

		return fc2_act

def create_tfslim_model(input_tensor, keep_prob_tensor):
	with tf.variable_scope('my_tfslim_scope', reuse=tf.AUTO_REUSE):  # Applied variables and operations.
	#with tf.name_scope('my_tfslim_scope'):  # Applied operations only.
		image_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])

		conv1_act = slim.conv2d(image_tensor, 32, [5, 5], activation_fn=tf.nn.relu)
		conv1_pool = slim.max_pool2d(conv1_act, [2, 2])

		conv2_act = slim.conv2d(conv1_pool, 64, [5, 5], activation_fn=tf.nn.relu)
		conv2_pool = slim.max_pool2d(conv2_act, [2, 2])

		conv2_flat = slim.flatten(conv2_pool)

		fc1_act = slim.fully_connected(conv2_flat, 1024, activation_fn=tf.nn.relu)
		fc1_dropout = slim.dropout(fc1_act, keep_prob_tensor)

		fc2_act = slim.fully_connected(fc1_dropout, 10, activation_fn=tf.nn.softmax)
	
		return fc2_act

def create_tfslim_named_model(input_tensor, keep_prob_tensor):
	with tf.variable_scope('my_tfslim_named_scope', reuse=tf.AUTO_REUSE):  # Applied variables and operations.
	#with tf.name_scope('my_tfslim_named_scope'):  # Applied operations only.
		image_tensor = tf.reshape(input_tensor, [-1, 28, 28, 1])

		conv1_act = slim.conv2d(image_tensor, 32, [5, 5], activation_fn=tf.nn.relu, scope='conv1')
		conv1_pool = slim.max_pool2d(conv1_act, [2, 2], scope='conv1')  # The scope 'conv1' is changed into 'conv1_1'.

		conv2_act = slim.conv2d(conv1_pool, 64, [5, 5], activation_fn=tf.nn.relu, scope='conv2')
		conv2_pool = slim.max_pool2d(conv2_act, [2, 2], scope='conv2')  # The scope 'conv2' is changed into 'conv2_1'.

		conv2_flat = slim.flatten(conv2_pool, scope='conv2')  # The scope 'conv2' is changed into 'conv2_2'.

		fc1_act = slim.fully_connected(conv2_flat, 1024, activation_fn=tf.nn.relu, scope='fc1')
		fc1_dropout = slim.dropout(fc1_act, keep_prob_tensor, scope='fc1')  # The scope 'fc1' is changed into 'fc1_1'.

		fc2_act = slim.fully_connected(fc1_dropout, 10, activation_fn=tf.nn.softmax, scope='fc2')
	
		return fc2_act

#%%------------------------------------------------------------------

image_ph = tf.placeholder(tf.float32, [None, 784], name='input_tensor')
label_ph = tf.placeholder(tf.float32, [None, 10], name='output_tensor')
keep_prob_ph = tf.placeholder(tf.float32)

tf_model_output_tensor = create_tf_model(image_ph, keep_prob_ph)
tf_named_model_output_tensor = create_tf_named_model(image_ph, keep_prob_ph)
tfslim_model_output_tensor = create_tfslim_model(image_ph, keep_prob_ph)
tfslim_named_model_output_tensor = create_tfslim_named_model(image_ph, keep_prob_ph)

model_output_tensor = tf_model_output_tensor
#model_output_tensor = tf_named_model_output_tensor
#model_output_tensor = tfslim_model_output_tensor
#model_output_tensor = tfslim_named_model_output_tensor

#%%------------------------------------------------------------------

cross_entropy = -tf.reduce_sum(label_ph * tf.log(model_output_tensor))
correct_prediction = tf.equal(tf.argmax(model_output_tensor, -1), tf.argmax(label_ph, -1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#%%------------------------------------------------------------------

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

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
# Get tensors from their names.

graph = sess.graph

if model_output_tensor is tf_model_output_tensor:
	conv1_preact = graph.get_tensor_by_name('my_tf_scope/add:0')
	conv1_act = graph.get_tensor_by_name('my_tf_scope/Relu:0')
	conv1_pool = graph.get_tensor_by_name('my_tf_scope/MaxPool:0')

	conv2_preact = graph.get_tensor_by_name('my_tf_scope/add_1:0')
	conv2_act = graph.get_tensor_by_name('my_tf_scope/Relu_1:0')
	conv2_pool = graph.get_tensor_by_name('my_tf_scope/MaxPool_1:0')

	fc1_preact = graph.get_tensor_by_name('my_tf_scope/add_2:0')
	fc1_act = graph.get_tensor_by_name('my_tf_scope/Relu_2:0')
	fc1_dropout = graph.get_tensor_by_name('my_tf_scope/dropout/mul:0')  # ???

	fc2_preact = graph.get_tensor_by_name('my_tf_scope/add_3:0')
	fc2_act = graph.get_tensor_by_name('my_tf_scope/Softmax:0')
elif model_output_tensor is tf_named_model_output_tensor:
	conv1_preact = graph.get_tensor_by_name('my_tf_named_scope/conv1/add:0')
	conv1_act = graph.get_tensor_by_name('my_tf_named_scope/conv1/activations:0')
	conv1_pool = graph.get_tensor_by_name('my_tf_named_scope/conv1/pooling:0')

	conv2_preact = graph.get_tensor_by_name('my_tf_named_scope/conv2/add:0')
	conv2_act = graph.get_tensor_by_name('my_tf_named_scope/conv2/activations:0')
	conv2_pool = graph.get_tensor_by_name('my_tf_named_scope/conv2/pooling:0')

	fc1_preact = graph.get_tensor_by_name('my_tf_named_scope/fc1/add:0')
	fc1_act = graph.get_tensor_by_name('my_tf_named_scope/fc1/activations:0')
	fc1_dropout = graph.get_tensor_by_name('my_tf_named_scope/fc1/dropout/mul:0')  # ???

	fc2_preact = graph.get_tensor_by_name('my_tf_named_scope/fc2/add:0')
	fc2_act = graph.get_tensor_by_name('my_tf_named_scope/fc2/activations:0')
elif model_output_tensor is tfslim_model_output_tensor:
	conv1_preact = graph.get_tensor_by_name('my_tfslim_scope/Conv/BiasAdd:0')
	conv1_act = graph.get_tensor_by_name('my_tfslim_scope/Conv/Relu:0')
	conv1_pool = graph.get_tensor_by_name('my_tfslim_scope/MaxPool2D/MaxPool:0')

	conv2_preact = graph.get_tensor_by_name('my_tfslim_scope/Conv_1/BiasAdd:0')
	conv2_act = graph.get_tensor_by_name('my_tfslim_scope/Conv_1/Relu:0')
	conv2_pool = graph.get_tensor_by_name('my_tfslim_scope/MaxPool2D_1/MaxPool:0')

	fc1_preact = graph.get_tensor_by_name('my_tfslim_scope/fully_connected/BiasAdd:0')
	fc1_act = graph.get_tensor_by_name('my_tfslim_scope/fully_connected/Relu:0')
	fc1_dropout = graph.get_tensor_by_name('my_tfslim_scope/Dropout/dropout/mul:0')  # ???

	fc2_preact = graph.get_tensor_by_name('my_tfslim_scope/fully_connected_1/BiasAdd:0')
	fc2_act = graph.get_tensor_by_name('my_tfslim_scope/fully_connected_1/Softmax:0')
elif model_output_tensor is tfslim_named_model_output_tensor:
	conv1_preact = graph.get_tensor_by_name('my_tfslim_named_scope/conv1/BiasAdd:0')
	conv1_act = graph.get_tensor_by_name('my_tfslim_named_scope/conv1/Relu:0')
	conv1_pool = graph.get_tensor_by_name('my_tfslim_named_scope/conv1_1/MaxPool:0')

	conv2_preact = graph.get_tensor_by_name('my_tfslim_named_scope/conv2/BiasAdd:0')
	conv2_act = graph.get_tensor_by_name('my_tfslim_named_scope/conv2/Relu:0')
	conv2_pool = graph.get_tensor_by_name('my_tfslim_named_scope/conv2_1/MaxPool:0')

	fc1_preact = graph.get_tensor_by_name('my_tfslim_named_scope/fc1/BiasAdd:0')
	fc1_act = graph.get_tensor_by_name('my_tfslim_named_scope/fc1/Relu:0')
	fc1_dropout = graph.get_tensor_by_name('my_tfslim_named_scope/fc1_1/dropout/mul:0')  # ???

	fc2_preact = graph.get_tensor_by_name('my_tfslim_named_scope/fc2/BiasAdd:0')
	fc2_act = graph.get_tensor_by_name('my_tfslim_named_scope/fc2/Softmax:0')
else:
	assert False, 'Invalid model type.'

#%%------------------------------------------------------------------
# Visualize activation.

# REF [function] >> plot_conv_filters() in ./tensorflow_activation_visualization_1.py.
def plot_conv_filters(units):
	filters = units.shape[3]
	plt.figure(1, figsize=(20, 20))
	n_columns = 6
	n_rows = math.ceil(filters / n_columns) + 1
	for i in range(filters):
		plt.subplot(n_rows, n_columns, i + 1)
		plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation='nearest', cmap='gray')

# REF [function] >> visual_activations() in ./tensorflow_activation_visualization_1.py.
def visual_activations(layer, stimuli):
	units = sess.run(layer, feed_dict={image_ph: np.reshape(stimuli, [1, 784], order='F'), keep_prob_ph: 1.0})  # units -> numpy.array.
	plot_conv_filters(units)

imageToUse = mnist.test.images[0]
plt.imshow(np.reshape(imageToUse, [28, 28]), interpolation='nearest', cmap='gray')

visual_activations(conv1_preact, imageToUse)
visual_activations(conv1_act, imageToUse)
visual_activations(conv1_pool, imageToUse)
visual_activations(conv2_preact, imageToUse)
visual_activations(conv2_act, imageToUse)
visual_activations(conv2_pool, imageToUse)
