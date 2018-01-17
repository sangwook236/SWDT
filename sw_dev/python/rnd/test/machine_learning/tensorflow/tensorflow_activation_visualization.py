# REF [site] >> https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4

import numpy as np 
import matplotlib as mp
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

#%%------------------------------------------------------------------

if 'posix' == os.name:
	#data_home_dir_path = '/home/sangwook/my_dataset'
	data_home_dir_path = '/home/HDD1/sangwook/my_dataset'
else:
	data_home_dir_path = 'D:/dataset'

mnist = input_data.read_data_sets(data_home_dir_path + '/pattern_recognition/mnist/0_original', one_hot=True)

#%%------------------------------------------------------------------

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#%%------------------------------------------------------------------

tf.reset_default_graph()

x_image_ph = tf.placeholder(tf.float32, [None, 784], name='x_tensor')
y_true_ph = tf.placeholder(tf.float32, [None, 10], name='y_true')
keep_prob = tf.placeholder(tf.float32)

x_tensor = tf.reshape(x_image_ph, [-1, 28, 28, 1])

conv1_W = weight_variable([5, 5, 1, 32])
conv1_b = bias_variable([32])
conv1_hidden = tf.nn.conv2d(input=x_tensor, filter=conv1_W, strides=[1, 2, 2, 1], padding='SAME') + conv1_b
conv1_hidden = tf.nn.relu(conv1_hidden)
conv1_pool = tf.nn.max_pool(conv1_hidden, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

conv2_W = weight_variable([5, 5, 32, 64])
conv2_b = bias_variable([64])
conv2_hidden = tf.nn.conv2d(input=conv1_pool, filter=conv2_W, strides=[1, 2, 2, 1], padding='SAME') + conv2_b
conv2_hidden = tf.nn.relu(conv2_hidden)
conv2_pool = tf.nn.max_pool(conv2_hidden, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

conv2_flat = tf.reshape(conv2_pool, [-1, 7 * 7 * 64])

fc1_W = weight_variable([7 * 7 * 64, 1024])
fc1_b = bias_variable([1024])
fc1_hidden = tf.nn.relu(tf.matmul(conv2_flat, fc1_W) + fc1_b)
fc1_drop = tf.nn.dropout(fc1_hidden, keep_prob)

fc2_W = weight_variable([1024, 10])
fc2_b = bias_variable([10])
y_pred = tf.nn.softmax(tf.matmul(fc1_drop, fc2_W) + fc2_b)

conv1_hidden = slim.conv2d(x_tensor, 5, [5, 5])
conv1_pool = slim.max_pool2d(conv1_hidden, [2, 2])
conv2_hidden = slim.conv2d(conv1_pool, 5, [5, 5])
conv2_pool = slim.max_pool2d(conv2_hidden, [2, 2])
conv3_hidden = slim.conv2d(conv2_pool, 20, [5, 5])
conv3_hidden = slim.dropout(conv3_hidden, keep_prob)
y_pred = slim.fully_connected(slim.flatten(conv3_hidden), 10, activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(y_true_ph * tf.log(y_pred))
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true_ph, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#%%------------------------------------------------------------------

batch_size = 50

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1001):
	batch = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x_image_ph: batch[0], y_true_ph: batch[1], keep_prob: 0.5})
	if i % 100 == 0 and i != 0:
		trainAccuracy = sess.run(accuracy, feed_dict={x_image_ph: batch[0], y_true_ph: batch[1], keep_prob: 1.0})
		print('Step %d, training accuracy %g' % (i, trainAccuracy))

testAccuracy = sess.run(accuracy, feed_dict={x_image_ph: mnist.test.images, y_true_ph: mnist.test.labels, keep_prob: 1.0})
print('Test accuracy %g' % (testAccuracy))

#%%------------------------------------------------------------------
# Visualize activation.

def plotNNFilter(units):
	filters = units.shape[3]
	plt.figure(1, figsize=(20, 20))
	n_columns = 6
	n_rows = math.ceil(filters / n_columns) + 1
	for i in range(filters):
		plt.subplot(n_rows, n_columns, i + 1)
		plt.title('Filter ' + str(i))
		plt.imshow(units[0,:,:,i], interpolation='nearest', cmap='gray')

def getActivations(layer, stimuli):
	units = sess.run(layer, feed_dict={x_image_ph: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})  # units -> numpy.array.
	plotNNFilter(units)

imageToUse = mnist.test.images[0]
plt.imshow(np.reshape(imageToUse, [28, 28]), interpolation='nearest', cmap='gray')

getActivations(conv1_hidden, imageToUse)
getActivations(conv2_hidden, imageToUse)
getActivations(conv3_hidden, imageToUse)
