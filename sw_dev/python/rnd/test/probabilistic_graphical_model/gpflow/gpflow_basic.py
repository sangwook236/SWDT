#!/usr/bin/env python

import gpflow
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/models.html
def handle_model_example():
	with gpflow.defer_build():
		X = np.random.rand(20, 1)
		Y = np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(20,1) * 0.01
		model = gpflow.models.GPR(X, Y, kern=gpflow.kernels.Matern32(1) + gpflow.kernels.Linear(1))

	# View, get and set parameters.
	print(model)
	print(model.as_pandas_table())
	print(model.likelihood.as_pandas_table())

	model.kern.kernels[0].lengthscales = 0.5
	model.likelihood.variance = 0.01
	print(model.as_pandas_table())

	# Constraints and trainable variables.
	print(model.read_trainables())

	model.kern.kernels[0].lengthscales.transform = gpflow.transforms.Exp()
	print(model.read_trainables())

	#model.kern.kernels[1].variance.trainable = False
	#print(model.as_pandas_table())
	#print(model.read_trainables())

	# Priors.
	model.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2, 3)
	print(model.as_pandas_table())

	# Optimization.
	model.compile()
	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model)

class LinearMulticlass(gpflow.models.Model):
	def __init__(self, X, Y, name=None):
		super().__init__(name=name)

		self.X = X.copy()  # X is a numpy array of inputs.
		self.Y = Y.copy()  # Y is a 1-of-K representation of the labels.

		self.num_data, self.input_dim = X.shape
		_, self.num_classes = Y.shape

		# Make some parameters.
		self.W = gpflow.Param(np.random.randn(self.input_dim, self.num_classes))
		self.b = gpflow.Param(np.random.randn(self.num_classes))

	@gpflow.params_as_tensors
	def _build_likelihood(self):
		# Param variables are used as Tensorflow arrays.
		p = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
		return tf.reduce_sum(tf.log(p) * self.Y)

# REF [site] >> https://gpflow.readthedocs.io/en/latest/notebooks/models.html
def build_new_model_example():
	plt.style.use('ggplot')
	#%matplotlib inline

	X = np.vstack([
		np.random.randn(10, 2) + [2, 2],
		np.random.randn(10, 2) +[-2, 2],
		np.random.randn(10, 2) +[2, -2]
	])
	Y = np.repeat(np.eye(3), 10, 0)

	matplotlib.rcParams['figure.figsize'] = (12, 6)
	plt.scatter(X[:,0], X[:,1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)

	model = LinearMulticlass(X, Y)
	print(model.as_pandas_table())

	opt = gpflow.train.ScipyOptimizer()
	opt.minimize(model)
	print(model.as_pandas_table())

	xx, yy = np.mgrid[-4:4:200j, -4:4:200j]
	X_test = np.vstack([xx.flatten(), yy.flatten()]).T
	f_test = np.dot(X_test, model.W.read_value()) + model.b.read_value()
	p_test = np.exp(f_test)
	p_test /= p_test.sum(1)[:,None]

	for i in range(3):
		plt.contour(xx, yy, p_test[:,i].reshape(200, 200), [0.5], colors='k', linewidths=1)
	plt.scatter(X[:,0], X[:,1], 100, np.argmax(Y, 1), lw=2, cmap=plt.cm.viridis)

def main():
	#handle_model_example()
	build_new_model_example()

#%%------------------------------------------------------------------

# Usage:
#	python gpflow_basic.py

if '__main__' == __name__:
	main()
