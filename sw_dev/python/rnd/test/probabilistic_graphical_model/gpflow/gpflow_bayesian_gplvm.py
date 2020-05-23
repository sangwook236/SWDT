#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pods
import gpflow
from gpflow import kernels

#%matplotlib inline

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/GPLVM.html
def bayesian_gplvm_example():
	np.random.seed(42)

	pods.datasets.overide_manual_authorize = True  # Dont ask to authorize.
	gpflow.settings.numerics.quadrature = 'error'  # Throw error if quadrature is used for kernel expectations.

	# Data.
	data = pods.datasets.oil_100()
	Y = data['X']
	print('Number of points X Number of dimensions', Y.shape)
	data['citation']

	# Model construction.
	Q = 5
	M = 20  # Number of inducing pts.
	N = Y.shape[0]
	X_mean = gpflow.models.PCA_reduce(Y, Q)  # Initialise via PCA.
	Z = np.random.permutation(X_mean.copy())[:M]

	fHmmm = False
	if fHmmm:
		k = (kernels.RBF(3, ARD=True, active_dims=slice(0, 3)) + kernels.Linear(2, ARD=False, active_dims=slice(3, 5)))
	else:
		k = (kernels.RBF(3, ARD=True, active_dims=[0, 1, 2]) + kernels.Linear(2, ARD=False, active_dims=[3, 4]))

	m = gpflow.models.BayesianGPLVM(X_mean=X_mean, X_var=0.1 * np.ones((N, Q)), Y=Y, kern=k, M=M, Z=Z)
	m.likelihood.variance = 0.01

	opt = gpflow.train.ScipyOptimizer()
	m.compile()
	opt.minimize(m)#, options=dict(disp=True, maxiter=100))

	# Compute and sensitivity to input.
	#	Sensitivity is a measure of the importance of each latent dimension.
	kern = m.kern.kernels[0]
	sens = np.sqrt(kern.variance.read_value()) / kern.lengthscales.read_value()
	print(m.kern)
	print(sens)
	fig, ax = plt.subplots()
	ax.bar(np.arange(len(kern.lengthscales.read_value())), sens, 0.1, color='y')
	ax.set_title('Sensitivity to latent inputs')

	# Plot vs PCA.
	XPCAplot = gpflow.models.PCA_reduce(data['X'], 2)
	f, ax = plt.subplots(1, 2, figsize=(10, 6))
	labels = data['Y'].argmax(axis=1)
	colors = cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))

	for i, c in zip(np.unique(labels), colors):
		ax[0].scatter(XPCAplot[labels==i,0], XPCAplot[labels==i,1], color=c, label=i)
		ax[0].set_title('PCA')
		ax[1].scatter(m.X_mean.read_value()[labels==i,1], m.X_mean.read_value()[labels==i,2], color=c, label=i)
		ax[1].set_title('Bayesian GPLVM')

def main():
	bayesian_gplvm_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
