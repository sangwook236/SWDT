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

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784], name='x-in')
true_y = tf.placeholder(tf.float32, [None, 10], name='y-in')
keep_prob = tf.placeholder('float')

x_image = tf.reshape(x,[-1, 28, 28, 1])
hidden_1 = slim.conv2d(x_image, 5, [5, 5])
pool_1 = slim.max_pool2d(hidden_1, [2, 2])
hidden_2 = slim.conv2d(pool_1, 5, [5, 5])
pool_2 = slim.max_pool2d(hidden_2, [2, 2])
hidden_3 = slim.conv2d(pool_2, 20, [5, 5])
hidden_3 = slim.dropout(hidden_3, keep_prob)
out_y = slim.fully_connected(slim.flatten(hidden_3), 10, activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(true_y * tf.log(out_y))
correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(true_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#%%------------------------------------------------------------------

batch_size = 50

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1001):
	batch = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 0.5})
	if i % 100 == 0 and i != 0:
		trainAccuracy = sess.run(accuracy, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 1.0})
		print('Step %d, training accuracy %g' % (i, trainAccuracy))

testAccuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, true_y: mnist.test.labels, keep_prob: 1.0})
print('Test accuracy %g' % (testAccuracy))

#%%------------------------------------------------------------------
# Visualize layer activation.

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
	units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [1, 784], order='F'), keep_prob: 1.0})  # units -> numpy.array.
	plotNNFilter(units)

imageToUse = mnist.test.images[0]
plt.imshow(np.reshape(imageToUse, [28, 28]), interpolation='nearest', cmap='gray')

getActivations(hidden_1, imageToUse)
getActivations(hidden_2, imageToUse)
getActivations(hidden_3, imageToUse)
