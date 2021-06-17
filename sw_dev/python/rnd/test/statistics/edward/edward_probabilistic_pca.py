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

def build_toy_dataset(N, D, K, sigma=1):
	x_train = np.zeros((D, N))
	w = np.random.normal(0.0, 2.0, size=(D, K))
	z = np.random.normal(0.0, 1.0, size=(K, N))
	mean = np.dot(w, z)
	for d in range(D):
		for n in range(N):
			x_train[d, n] = np.random.normal(mean[d, n], sigma)

	print('True principal axes:')
	print(w)
	return x_train

# REF [site] >> http://edwardlib.org/tutorials/probabilistic-pca
def probabilistic_pca_example():
	ed.set_seed(142)

	N = 5000  # Number of data points.
	D = 2  # Data dimensionality.
	K = 1  # Latent dimensionality.

	x_train = build_toy_dataset(N, D, K)

	plt.scatter(x_train[0, :], x_train[1, :], color='blue', alpha=0.1)
	plt.axis([-10, 10, -10, 10])
	plt.title('Simulated data set')
	plt.show()

	#--------------------
	# Model.
	w = Normal(loc=tf.zeros([D, K]), scale=2.0 * tf.ones([D, K]))
	z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K]))
	x = Normal(loc=tf.matmul(w, z, transpose_b=True), scale=tf.ones([D, N]))

	#--------------------
	# Inference.
	qw = Normal(loc=tf.get_variable('qw/loc', [D, K]), scale=tf.nn.softplus(tf.get_variable('qw/scale', [D, K])))
	qz = Normal(loc=tf.get_variable('qz/loc', [N, K]), scale=tf.nn.softplus(tf.get_variable('qz/scale', [N, K])))

	inference = ed.KLqp({w: qw, z: qz}, data={x: x_train})
	inference.run(n_iter=500, n_print=100, n_samples=10)

	#--------------------
	# Criticism.
	sess = ed.get_session()
	print('Inferred principal axes:')
	print(sess.run(qw.mean()))

	# Build and then generate data from the posterior predictive distribution.
	x_post = ed.copy(x, {w: qw, z: qz})
	x_gen = sess.run(x_post)

	plt.scatter(x_gen[0, :], x_gen[1, :], color='red', alpha=0.1)
	plt.axis([-10, 10, -10, 10])
	plt.title('Data generated from model')
	plt.show()

def main():
	probabilistic_pca_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
