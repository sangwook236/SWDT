#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def build_toy_dataset(N=50, noise_std=0.1):
	x = np.linspace(-3, 3, num=N)
	y = np.cos(x) + np.random.normal(0, noise_std, size=N)
	x = x.astype(np.float32).reshape((N, 1))
	y = y.astype(np.float32)
	return x, y

def neural_network(x, W_0, W_1, b_0, b_1):
	h = tf.tanh(tf.matmul(x, W_0) + b_0)
	h = tf.matmul(h, W_1) + b_1
	return tf.reshape(h, [-1])

# REF [site] >> http://edwardlib.org/getting-started
def getting_started_example():
	# Simulate a toy dataset of 50 observations with a cosine relationship.
	ed.set_seed(42)

	N = 50  # Number of data points.
	D = 1  # Number of features.

	x_train, y_train = build_toy_dataset(N)

	#--------------------
	# Define a two-layer Bayesian neural network.
	W_0 = Normal(loc=tf.zeros([D, 2]), scale=tf.ones([D, 2]))
	W_1 = Normal(loc=tf.zeros([2, 1]), scale=tf.ones([2, 1]))
	b_0 = Normal(loc=tf.zeros(2), scale=tf.ones(2))
	b_1 = Normal(loc=tf.zeros(1), scale=tf.ones(1))

	x = x_train
	y = Normal(loc=neural_network(x, W_0, W_1, b_0, b_1), scale=0.1 * tf.ones(N))

	#--------------------
	# Make inferences about the model from data. 
	# We will use variational inference.
	# Specify a normal approximation over the weights and biases.
	# Defining tf.get_variable allows the variational factors' parameters to vary. They are initialized randomly.
	# The standard deviation parameters are constrained to be greater than zero according to a softplus transformation.
	qW_0 = Normal(loc=tf.get_variable('qW_0/loc', [D, 2]), scale=tf.nn.softplus(tf.get_variable('qW_0/scale', [D, 2])))
	qW_1 = Normal(loc=tf.get_variable('qW_1/loc', [2, 1]), scale=tf.nn.softplus(tf.get_variable('qW_1/scale', [2, 1])))
	qb_0 = Normal(loc=tf.get_variable('qb_0/loc', [2]), scale=tf.nn.softplus(tf.get_variable('qb_0/scale', [2])))
	qb_1 = Normal(loc=tf.get_variable('qb_1/loc', [1]), scale=tf.nn.softplus(tf.get_variable('qb_1/scale', [1])))

	# Sample functions from variational model to visualize fits.
	rs = np.random.RandomState(0)
	inputs = np.linspace(-5, 5, num=400, dtype=np.float32)
	x = tf.expand_dims(inputs, 1)
	mus = tf.stack([neural_network(x, qW_0.sample(), qW_1.sample(), qb_0.sample(), qb_1.sample()) for _ in range(10)])

	# First Visualization (prior).
	sess = ed.get_session()
	tf.global_variables_initializer().run()
	outputs = mus.eval()

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	ax.set_title('Iteration: 0')
	ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
	ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='prior draws')
	ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
	ax.set_xlim([-5, 5])
	ax.set_ylim([-2, 2])
	ax.legend()
	plt.show()

	#--------------------
	# Run variational inference with the Kullback-Leibler divergence in order to infer the model's latent variables with the given data.
	# We specify 1000 iterations.
	inference = ed.KLqp({W_0: qW_0, b_0: qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_train})
	inference.run(n_iter=1000, n_samples=5)

	#--------------------
	# Criticize the model fit.
	# Bayesian neural networks define a distribution over neural networks, so we can perform a graphical check.
	# Draw neural networks from the inferred model and visualize how well it fits the data.

	# SECOND VISUALIZATION (posterior)
	outputs = mus.eval()

	fig = plt.figure(figsize=(10, 6))
	ax = fig.add_subplot(111)
	ax.set_title('Iteration: 1000')
	ax.plot(x_train, y_train, 'ks', alpha=0.5, label='(x, y)')
	ax.plot(inputs, outputs[0].T, 'r', lw=2, alpha=0.5, label='posterior draws')
	ax.plot(inputs, outputs[1:].T, 'r', lw=2, alpha=0.5)
	ax.set_xlim([-5, 5])
	ax.set_ylim([-2, 2])
	ax.legend()
	plt.show()

def main():
	getting_started_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
