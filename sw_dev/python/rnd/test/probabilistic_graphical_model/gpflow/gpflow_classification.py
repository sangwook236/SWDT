#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, csv
import numpy as np
from scipy.cluster.vq import kmeans
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import gpflow

def gridParams():
	mins = [-3.25, -2.85]
	maxs = [3.65, 3.4]
	nGrid = 50
	xspaced = np.linspace(mins[0], maxs[0], nGrid)
	yspaced = np.linspace(mins[1], maxs[1], nGrid)
	xx, yy = np.meshgrid(xspaced, yspaced)
	Xplot = np.vstack((xx.flatten(),yy.flatten())).T
	return mins, maxs, xx, yy, Xplot

def plot(m, ax):
	col1 = '#0172B2'
	col2 = '#CC6600'
	mins, maxs, xx, yy, Xplot = gridParams()
	p = m.predict_y(Xplot)[0]
	ax.plot(Xtrain[:,0][Ytrain[:,0] == 1], Xtrain[:,1][Ytrain[:,0] == 1], 'o', color=col1, mew=0, alpha=0.5)
	ax.plot(Xtrain[:,0][Ytrain[:,0] == 0], Xtrain[:,1][Ytrain[:,0] == 0], 'o', color=col2, mew=0, alpha=0.5)
	if hasattr(m, 'feat') and hasattr(m.feat, 'Z'):
		Z = m.feature.Z.read_value()
		ax.plot(Z[:,0], Z[:,1], 'ko', mew=0, ms=4)
		ax.set_title('m={}'.format(Z.shape[0]))
	else:
		ax.set_title('full')
	ax.contour(xx, yy, p.reshape(*xx.shape), [0.5], colors='k', linewidths=1.8, zorder=100)

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/classification.html
def classification_example():
	%matplotlib inline
	plt.style.use('ggplot')

	Xtrain = np.loadtxt('dataset/banana_X_train.csv', delimiter=',')
	Ytrain = np.loadtxt('dataset/banana_Y_train.csv', delimiter=',').reshape(-1, 1)

	# Setup the experiment and plotting.
	Ms = [4, 8, 16, 32, 64]

	# Run sparse classification with increasing number of inducing points.
	models = []
	for index, num_inducing in enumerate(Ms):
		# kmeans for selecting Z.
		Z = kmeans(Xtrain, num_inducing)[0]

		model = gpflow.models.SVGP(
			Xtrain, Ytrain,
			kern=gpflow.kernels.RBF(2),
			likelihood=gpflow.likelihoods.Bernoulli(),
			Z=Z
		)
		# Initially fix the hyperparameters.
		model.feature.set_trainable(False)
		gpflow.train.ScipyOptimizer().minimize(model, maxiter=20)

		# Unfix the hyperparameters.
		model.feature.set_trainable(True)
		gpflow.train.ScipyOptimizer(options=dict(maxiter=200)).minimize(model)
		models.append(model)

		# Run variational approximation without sparsity.
		# Be aware that this is much slower for big datasets, but relatively quick here.
		m = gpflow.models.VGP(
			Xtrain, Ytrain,
			kern=gpflow.kernels.RBF(2),
			likelihood=gpflow.likelihoods.Bernoulli()
		)
		gpflow.train.ScipyOptimizer().minimize(m, maxiter=2000)
		models.append(m)

	# Make plots.
	fig, axes = plt.subplots(1, len(models), figsize=(12.5, 2.5), sharex=True, sharey=True)
	for i, model in enumerate(models):
		plot(model, axes[i])
		axes[i].set_yticks([])
		axes[i].set_xticks([])

def main():
	classification_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
