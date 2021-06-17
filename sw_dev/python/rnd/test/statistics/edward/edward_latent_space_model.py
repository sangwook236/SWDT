#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal, Poisson
from observations import celegans

# REF [site] >> http://edwardlib.org/tutorials/latent-space-models
def latent_space_model_example():
	x_train = celegans('~/data')

	#--------------------
	N = x_train.shape[0]  # Number of data points.
	K = 3  # Latent dimensionality.

	z = Normal(loc=tf.zeros([N, K]), scale=tf.ones([N, K]))

	# Calculate N x N distance matrix.
	# 1. Create a vector, [||z_1||^2, ||z_2||^2, ..., ||z_N||^2], and tile it to create N identical rows.
	xp = tf.tile(tf.reduce_sum(tf.pow(z, 2), 1, keep_dims=True), [1, N])
	# 2. Create a N x N matrix where entry (i, j) is ||z_i||^2 + ||z_j||^2 - 2 z_i^T z_j.
	xp = xp + tf.transpose(xp) - 2 * tf.matmul(z, z, transpose_b=True)
	# 3. Invert the pairwise distances and make rate along diagonals to be close to zero.
	xp = 1.0 / tf.sqrt(xp + tf.diag(tf.zeros(N) + 1e3))

	x = Poisson(rate=xp)

	#--------------------
	if True:
		# Maximum a posteriori (MAP) estimation is simple in Edward.
		inference = ed.MAP([z], data={x: x_train})
	else:
		# One could run variational inference.
		qz = Normal(loc=tf.get_variable('qz/loc', [N * K]), scale=tf.nn.softplus(tf.get_variable('qz/scale', [N * K])))
		inference = ed.KLqp({z: qz}, data={x: x_train})
	def main():
		latent_space_model_example()

	inference.run(n_iter=2500)

def main():
	latent_space_model_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
