#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import seaborn as sns
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Mixture, Normal
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
	"""Plots the mixture of Normal models to axis=ax comp=True plots all
	components of mixture model
	"""
	x = np.linspace(-10.5, 10.5, 250)
	final = np.zeros_like(x)
	for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
		temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
		final = final + temp
		if comp:
			ax.plot(x, temp, label='Normal ' + str(i))
	ax.plot(x, final, label='Mixture of Normals ' + label)
	ax.legend(fontsize=13)

def sample_from_mixture(x, pred_weights, pred_means, pred_std, amount):
	"""Draws samples from mixture model.

	Returns 2 d array with input X and sample from prediction of mixture model.
	"""
	samples = np.zeros((amount, 2))
	n_mix = len(pred_weights[0])
	to_choose_from = np.arange(n_mix)
	for j, (weights, means, std_devs) in enumerate(zip(pred_weights, pred_means, pred_std)):
		index = np.random.choice(to_choose_from, p=weights)
		samples[j, 1] = np.random.normal(means[index], std_devs[index], size=1)
		samples[j, 0] = x[j]
		if j == amount - 1:
			break
	return samples

def build_toy_dataset(N):
	y_data = np.random.uniform(-10.5, 10.5, N)
	r_data = np.random.normal(size=N)  # Random noise.
	x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
	x_data = x_data.reshape((N, 1))
	return train_test_split(x_data, y_data, random_state=42)

def neural_network(X, K):
	"""loc, scale, logits = NN(x; theta)"""
	# 2 hidden layers with 15 hidden units.
	net = tf.layers.dense(X, 15, activation=tf.nn.relu)
	net = tf.layers.dense(net, 15, activation=tf.nn.relu)
	locs = tf.layers.dense(net, K, activation=None)
	scales = tf.layers.dense(net, K, activation=tf.exp)
	logits = tf.layers.dense(net, K, activation=None)
	return locs, scales, logits

# REF [site] >> http://edwardlib.org/tutorials/mixture-density-network
def mixture_density_network_example():
	ed.set_seed(42)

	N = 5000  # Number of data points.
	D = 1  # Number of features.
	K = 20  # Number of mixture components.

	X_train, X_test, y_train, y_test = build_toy_dataset(N)
	print('Size of features in training data: {}'.format(X_train.shape))
	print('Size of output in training data: {}'.format(y_train.shape))
	print('Size of features in test data: {}'.format(X_test.shape))
	print('Size of output in test data: {}'.format(y_test.shape))
	sns.regplot(X_train, y_train, fit_reg=False)
	plt.show()

	#--------------------
	X_ph = tf.placeholder(tf.float32, [None, D])
	y_ph = tf.placeholder(tf.float32, [None])

	# We use a mixture of 20 normal distributions parameterized by a feedforward network.
	#	The membership probabilities and per-component mean and standard deviation are given by the output of a feedforward network.
	# We use tf.layers to construct neural networks.
	# We specify a three-layer network with 15 hidden units for each hidden layer.

	locs, scales, logits = neural_network(X_ph, K)
	cat = Categorical(logits=logits)
	components = [Normal(loc=loc, scale=scale) for loc, scale in zip(tf.unstack(tf.transpose(locs)), tf.unstack(tf.transpose(scales)))]
	y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))
	# Note: A bug exists in Mixture which prevents samples from it to have a shape of [None].
	# For now fix it using the value argument, as sampling is not necessary for MAP estimation anyways.

	#--------------------
	# We use MAP estimation, passing in the model and data set.

	# There are no latent variables to infer.
	# Thus inference is concerned with only training model parameters, which are baked into how we specify the neural networks.
	inference = ed.MAP(data={y: y_ph})
	optimizer = tf.train.AdamOptimizer(5e-3)
	inference.initialize(optimizer=optimizer, var_list=tf.trainable_variables())

	# we will manually control the inference and how data is passed into it at each step.
	# Initialize the algorithm and the TensorFlow variables.
	sess = ed.get_session()
	tf.global_variables_initializer().run()

	# Now we train the MDN by calling inference.update(), passing in the data.
	# The quantity inference.loss is the loss function (negative log-likelihood) at that step of inference.
	# We also report the loss function on test data by calling inference.loss and where we feed test data to the TensorFlow placeholders instead of training data.
	# We keep track of the losses under train_loss and test_loss.
	n_epoch = 1000
	train_loss = np.zeros(n_epoch)
	test_loss = np.zeros(n_epoch)
	for i in range(n_epoch):
		info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
		train_loss[i] = info_dict['loss']
		test_loss[i] = sess.run(inference.loss, feed_dict={X_ph: X_test, y_ph: y_test})
		inference.print_progress(info_dict)

	#--------------------
	# After training for a number of iterations, we get out the predictions we are interested in from the model: the predicted mixture weights, cluster means, and cluster standard deviations.
	# To do this, we fetch their values from session, feeding test data X_test to the placeholder X_ph.
	pred_weights, pred_means, pred_std = sess.run([tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})

	# Let's plot the log-likelihood of the training and test data as functions of the training epoch.
	# The quantity inference.loss is the total log-likelihood, not the loss per data point.
	# Below we plot the per-data point log-likelihood by dividing by the size of the train and test data respectively.
	fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
	plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
	plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
	plt.legend(fontsize=20)
	plt.xlabel('Epoch', fontsize=15)
	plt.ylabel('Log-likelihood', fontsize=15)
	plt.show()

	#--------------------
	# Criticism.
	# Note that as this is an inverse problem we can't get the answer correct, but we can hope that the truth lies in area where the model has high probability.
	obj = [0, 4, 6]
	fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 6))

	plot_normal_mix(pred_weights[obj][0], pred_means[obj][0], pred_std[obj][0], axes[0], comp=False)
	axes[0].axvline(x=y_test[obj][0], color='black', alpha=0.5)

	plot_normal_mix(pred_weights[obj][2], pred_means[obj][2], pred_std[obj][2], axes[1], comp=False)
	axes[1].axvline(x=y_test[obj][2], color='black', alpha=0.5)

	plot_normal_mix(pred_weights[obj][1], pred_means[obj][1], pred_std[obj][1], axes[2], comp=False)
	axes[2].axvline(x=y_test[obj][1], color='black', alpha=0.5)
	plt.show()

	# We can check the ensemble by drawing samples of the prediction and plotting the density of those.
	a = sample_from_mixture(X_test, pred_weights, pred_means, pred_std, amount=len(X_test))
	sns.jointplot(a[:, 0], a[:, 1], kind='hex', color='#4CB391', ylim=(-10, 10), xlim=(-14, 14))
	plt.show()

def main():
	mixture_density_network_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
