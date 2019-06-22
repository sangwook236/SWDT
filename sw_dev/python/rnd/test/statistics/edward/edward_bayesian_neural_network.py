#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal

def build_toy_dataset(N=40, noise_std=0.1):
	D = 1
	X = np.concatenate([np.linspace(0, 2, num=N / 2), np.linspace(6, 8, num=N / 2)])
	y = np.cos(X) + np.random.normal(0, noise_std, size=N)
	X = (X - 4.0) / 4.0
	X = X.reshape((N, D))
	return X, y

def neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2):
	h = tf.tanh(tf.matmul(X, W_0) + b_0)
	h = tf.tanh(tf.matmul(h, W_1) + b_1)
	h = tf.matmul(h, W_2) + b_2
	return tf.reshape(h, [-1])

# REF [site] >> http://edwardlib.org/tutorials/bayesian-neural-network
def generative_adversarial_network_example():
	ed.set_seed(42)

	N = 40  # Number of data points.
	D = 1  # Number of features.

	X_train, y_train = build_toy_dataset(N)

	#--------------------
	# Model.
	with tf.name_scope('model'):
		W_0 = Normal(loc=tf.zeros([D, 10]), scale=tf.ones([D, 10]), name='W_0')
		W_1 = Normal(loc=tf.zeros([10, 10]), scale=tf.ones([10, 10]), name='W_1')
		W_2 = Normal(loc=tf.zeros([10, 1]), scale=tf.ones([10, 1]), name='W_2')
		b_0 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name='b_0')
		b_1 = Normal(loc=tf.zeros(10), scale=tf.ones(10), name='b_1')
		b_2 = Normal(loc=tf.zeros(1), scale=tf.ones(1), name='b_2')

		X = tf.placeholder(tf.float32, [N, D], name='X')
		y = Normal(loc=neural_network(X, W_0, W_1, W_2, b_0, b_1, b_2), scale=0.1 * tf.ones(N), name='y')

	#--------------------
	# Inference.
	with tf.variable_scope('posterior'):
		with tf.variable_scope('qW_0'):
			loc = tf.get_variable('loc', [D, 10])
			scale = tf.nn.softplus(tf.get_variable('scale', [D, 10]))
			qW_0 = Normal(loc=loc, scale=scale)
		with tf.variable_scope('qW_1'):
			loc = tf.get_variable('loc', [10, 10])
			scale = tf.nn.softplus(tf.get_variable('scale', [10, 10]))
			qW_1 = Normal(loc=loc, scale=scale)
		with tf.variable_scope('qW_2'):
			loc = tf.get_variable('loc', [10, 1])
			scale = tf.nn.softplus(tf.get_variable('scale', [10, 1]))
			qW_2 = Normal(loc=loc, scale=scale)
		with tf.variable_scope('qb_0'):
			loc = tf.get_variable('loc', [10])
			scale = tf.nn.softplus(tf.get_variable('scale', [10]))
			qb_0 = Normal(loc=loc, scale=scale)
		with tf.variable_scope('qb_1'):
			loc = tf.get_variable('loc', [10])
			scale = tf.nn.softplus(tf.get_variable('scale', [10]))
			qb_1 = Normal(loc=loc, scale=scale)
		with tf.variable_scope('qb_2'):
			loc = tf.get_variable('loc', [1])
			scale = tf.nn.softplus(tf.get_variable('scale', [1]))
			qb_2 = Normal(loc=loc, scale=scale)

	inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1,  W_2: qW_2, b_2: qb_2}, data={X: X_train, y: y_train})
	inference.run(logdir='log')

def main():
	generative_adversarial_network_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
