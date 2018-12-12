#!/usr/bin/env python

import sys, csv
import numpy as np
from scipy.cluster.vq import kmeans
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import gpflow

def plot(m):
	f = plt.figure(figsize=(12, 6))
	a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
	a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
	a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

	xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1,1)
	mu, var = m.predict_f(xx)
	mu, var = mu.copy(), var.copy()
	p, _ = m.predict_y(xx)

	a3.set_xticks([])
	a3.set_yticks([])

	a3.set_xticks([])
	a3.set_yticks([])

	for i in range(m.likelihood.num_classes):
		x = m.X.read_value()[m.Y.read_value().flatten() == i]
		points, = a3.plot(x, x * 0, '.')
		color = points.get_color()
		a1.plot(xx, mu[:,i], color=color, lw=2)
		a1.plot(xx, mu[:,i] + 2 * np.sqrt(var[:,i]), '--', color=color)
		a1.plot(xx, mu[:,i] - 2 * np.sqrt(var[:,i]), '--', color=color)
		a2.plot(xx, p[:,i], '-', color=color, lw=2)

	a2.set_ylim(-0.1, 1.1)
	a2.set_yticks([0, 1])
	a2.set_xticks([])

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/multiclass.html
def sparse_variational_gaussian_approximation_example():
	#%matplotlib inline
	plt.style.use('ggplot')

	np.random.seed(1)

	# Make a one dimensional classification problem.
	X = np.random.rand(100, 1)
	K = np.exp(-0.5 * np.square(X - X.T) / 0.01) + np.eye(100) * 1e-6
	f = np.dot(np.linalg.cholesky(K), np.random.randn(100, 3))
	Y = np.array(np.argmax(f, 1).reshape(-1, 1), dtype=float)

	plt.figure(figsize=(12,6))
	plt.plot(X, f, '.')

	model = gpflow.models.SVGP(
	    X, Y,
	    kern=gpflow.kernels.Matern32(1) + gpflow.kernels.White(1, variance=0.01),
	    likelihood=gpflow.likelihoods.MultiClass(3),
	    Z=X[::5].copy(), num_latent=3, whiten=True, q_diag=True
	)

	model.kern.kernels[1].variance.trainable = False
	model.feature.trainable = False
	print(model.as_pandas_table())

	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model)

	plot(model)

	print(model.kern.as_pandas_table())

def plot_from_samples(m, samples):
	f = plt.figure(figsize=(12, 6))
	a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
	a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
	a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

	xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1, 1)

	Fpred, Ypred = [], []
	for s in samples[100::10].iterrows():  # Burn 100, thin 10.
		m.assign(s[1])
		Ypred.append(m.predict_y(xx)[0])
		Fpred.append(m.predict_f_samples(xx, 1).squeeze())

	for i in range(m.likelihood.num_classes):
		x = m.X.read_value()[m.Y.read_value().flatten() == i]
		points, = a3.plot(x, x * 0, '.')
		color = points.get_color()
		for F in Fpred:
			a1.plot(xx, F[:,i], color=color, lw=0.2, alpha=1.0)
		for Y in Ypred:
			a2.plot(xx, Y[:,i], color=color, lw=0.5, alpha=1.0)

	a2.set_ylim(-0.1, 1.1)
	a2.set_yticks([0, 1])
	a2.set_xticks([])

	a3.set_xticks([])
	a3.set_yticks([])

def sparse_mcmc_example():
	#%matplotlib inline
	plt.style.use('ggplot')

	np.random.seed(1)

	# Make a one dimensional classification problem.
	X = np.random.rand(100, 1)
	K = np.exp(-0.5 * np.square(X - X.T) / 0.01) + np.eye(100) * 1e-6
	f = np.dot(np.linalg.cholesky(K), np.random.randn(100, 3))
	Y = np.array(np.argmax(f, 1).reshape(-1, 1), dtype=float)

	plt.figure(figsize=(12,  6))
	plt.plot(X, f, '.')

	with gpflow.defer_build():
		model = gpflow.models.SGPMC(
			X, Y,
			kern=gpflow.kernels.Matern32(1, lengthscales=0.1) + gpflow.kernels.White(1, variance=0.01),
			likelihood=gpflow.likelihoods.MultiClass(3),
			Z=X[::5].copy(), num_latent=3
		)
		model.kern.kernels[0].variance.prior = gpflow.priors.Gamma(1.0, 1.0)
		model.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(2.0, 2.0)
		model.kern.kernels[1].variance.trainables = False

	model.compile()
	print(model.as_pandas_table())

	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model, maxiter=10)
	print(model.kern.as_pandas_table())

	hmc = gpflow.train.HMC()
	samples = hmc.sample(model, num_samples=500, epsilon=0.04, lmax=15, logprobs=False)  # pands.DataFrame.
	#print('Columns =', samples.columns.values)

	plot_from_samples(model, samples)
	print(samples.head())

	_ = plt.hist(np.vstack(samples['SGPMC/kern/kernels/0/lengthscales']).flatten(), 50, density=True)
	plt.xlabel('lengthscale')

def main():
	sparse_variational_gaussian_approximation_example()
	sparse_mcmc_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
