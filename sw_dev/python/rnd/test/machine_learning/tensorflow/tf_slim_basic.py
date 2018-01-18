# REF [site] >> https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

import tensorflow.contrib.slim as slim
import tensorflow as tf

#%%------------------------------------------------------------------
# Variables.

# Model variables.
weights = slim.model_variable('weights',
	shape=[10, 10, 3 , 3],
	initializer=tf.truncated_normal_initializer(stddev=0.1),
	regularizer=slim.l2_regularizer(0.05),
	device='/CPU:0')
model_variables = slim.get_model_variables()

# Regular variables.
my_var = slim.variable('my_var',
	shape=[20, 1],
	initializer=tf.zeros_initializer())
regular_variables_and_model_variables = slim.get_variables()

"""
my_model_variable = CreateViaCustomCode()

# Letting TF-Slim know about the additional variable.
slim.add_model_variable(my_model_variable)
"""

#%%------------------------------------------------------------------
# Layers.

input_shape = (None, 224, 224, 3)
input_tensor = tf.placeholder(tf.float32, shape=input_shape)

"""
net = slim.conv2d(input_tensor, 256, [3, 3], scope='conv3_1')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')
"""
net = slim.repeat(input_tensor, 3, slim.conv2d, 256, [3, 3], scope='conv3')
net = slim.max_pool2d(net, [2, 2], scope='pool2')

"""
net = slim.fully_connected(net, 32, scope='fc/fc_1')
net = slim.fully_connected(net, 64, scope='fc/fc_2')
net = slim.fully_connected(net, 128, scope='fc/fc_3')
"""
slim.stack(net, slim.fully_connected, [32, 64, 128], scope='fc')

"""
net = slim.conv2d(net, 32, [3, 3], scope='core/core_1')
net = slim.conv2d(net, 32, [1, 1], scope='core/core_2')
net = slim.conv2d(net, 64, [3, 3], scope='core/core_3')
net = slim.conv2d(net, 64, [1, 1], scope='core/core_4')
"""
slim.stack(net, slim.conv2d, [(32, [3, 3]), (32, [1, 1]), (64, [3, 3]), (64, [1, 1])], scope='core')
