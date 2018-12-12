#!/usr/bin/env python

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib
import gpflow

def plotkernelsample(k, ax, xmin=-3, xmax=3):
	xx = np.linspace(xmin, xmax, 100)[:, None]
	K = k.compute_K_symm(xx)
	ax.plot(xx, np.random.multivariate_normal(np.zeros(100), K, 3).T)
	ax.set_title(k.__class__.__name__)

def plotkernelfunction(k, ax, xmin=-3, xmax=3, other=0):
	xx = np.linspace(xmin, xmax, 100)[:, None]
	#K = k.compute_K_symm(xx)
	ax.plot(xx, k.compute_K(xx, np.zeros((1, 1)) + other))
	ax.set_title(k.__class__.__name__ + ' k(x, %f)' % other)

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/kernels.html
def kernel_example():
	#%matplotlib inline
	plt.style.use('ggplot')

	# Kernel choices.
	f, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True, sharey=True)
	plotkernelsample(gpflow.kernels.Matern12(1), axes[0,0])
	plotkernelsample(gpflow.kernels.Matern32(1), axes[0,1])
	plotkernelsample(gpflow.kernels.Matern52(1), axes[0,2])
	plotkernelsample(gpflow.kernels.RBF(1), axes[0,3])
	plotkernelsample(gpflow.kernels.Constant(1), axes[1,0])
	plotkernelsample(gpflow.kernels.Linear(1), axes[1,1])
	plotkernelsample(gpflow.kernels.Cosine(1), axes[1,2])
	plotkernelsample(gpflow.kernels.Periodic(1), axes[1,3])
	axes[0,0].set_ylim(-3, 3)

	# Combine kernels.
	k1 = gpflow.kernels.Matern12(input_dim=1)
	k2 = gpflow.kernels.Linear(input_dim=1)

	k3 = k1 + k2
	k4 = k1 * k2

	# Kernels on multiple dimemensions.
	k = gpflow.kernels.Matern52(input_dim=5)
	print(k.as_pandas_table())

	k = gpflow.kernels.Matern52(input_dim=5, ARD=True)
	print(k.as_pandas_table())

	# Active dimensions.
	k1 = gpflow.kernels.Linear(1, active_dims=[0])
	k2 = gpflow.kernels.Matern52(1, active_dims=[1])
	k = k1 + k2
	print(k.as_pandas_table())

class Brownian(gpflow.kernels.Kernel):
	def __init__(self):
		super().__init__(input_dim=1, active_dims=[0])
		self.variance = gpflow.Param(1.0, transform=gpflow.transforms.positive)

	def K(self, X, X2=None):
		if X2 is None:
			X2 = X
		return self.variance * tf.minimum(X, tf.transpose(X2))

	def Kdiag(self, X):
		return self.variance * tf.reshape(X, (-1,))

def making_new_kernel_example():
	#%matplotlib inline
	plt.style.use('ggplot')

	X = np.random.rand(5, 1)
	Y = np.sin(X * 6) + np.random.randn(*X.shape) * 0.001

	k1 = Brownian()
	k2 = gpflow.kernels.Constant(1)
	k = k1 + k2
	print(k.as_pandas_table())

	model = gpflow.models.GPR(X, Y, kern=k)
	print(model.as_pandas_table())

	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model)
	print(model.kern.as_pandas_table())

	xx = np.linspace(0, 1.1, 100).reshape(100, 1)
	mean, var = model.predict_y(xx)
	plt.plot(X, Y, 'kx', mew=2)
	line, = plt.plot(xx, mean, lw=2)
	_ = plt.fill_between(xx[:,0], mean[:,0] - 2 * np.sqrt(var[:,0]), mean[:,0] + 2 * np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)

def main():
	kernel_example()
	making_new_kernel_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
