#!/usr/bin/env python

import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from IPython.display import display

# REF [site] >> http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/models_basic.ipynb
def basic_model_tutorial():
	%matplotlib inline
	%config InlineBackend.figure_format = 'svg'
	matplotlib.rcParams['figure.figsize'] = (8, 6)

	model = GPy.examples.regression.sparse_GP_regression_1D(plot=False, optimize=False)

	# Examine the model using print.
	print('Model =', model)
	print('Model RBF =', model.rbf)
	print('Model inducing inputs =', model.inducing_inputs)

	# Interact with parameters.
	model.inducing_inputs[0] = 1
	model.rbf.lengthscale = 0.2
	print('Model =', model)

	# Regular expressions.
	print(model['.*var'])
	model['.*var'] = 2.
	print(model['.*var'])
	model['.*var'] = [2., 3.]
	print(model['.*var'])
	# See all of the parameters of the model at once.
	print(model[''])

	# Set and fetch parameters 'parameter array'.
	model[:] = np.r_[[-4, -2, 0, 2, 4], [0.1, 2], [0.7]]
	print('Model =', model)

	model.inducing_inputs[2:, 0] = [1,3,5]
	print('Model inducing inputs =', model.inducing_inputs)

	# Get the model parameter's gradients.
	print('All gradients of the model:\n', model.gradient)
	print('Gradients of the rbf kernel:\n', model.rbf.gradient)

	model.optimize()
	print('Model gradient =\n', model.gradient)

	# Adjust the model's constraints.
	model.rbf.variance.unconstrain()
	print('Model =', model)
	model.unconstrain()
	print('Model =', model)

	model.inducing_inputs[0].fix()
	model.rbf.constrain_positive()
	print('Model =', model)
	model.unfix()
	print('Model =', model)

	# Optimize the model.
	model.Gaussian_noise.constrain_positive()
	model.rbf.constrain_positive()
	model.optimize()

	"""
	# Plot.
	fig = model.plot()

	GPy.plotting.change_plotting_library('plotly')
	fig = model.plot(plot_density=True)
	GPy.plotting.show(fig, filename='gpy_sparse_gp_example')
	"""

# REF [site] >> http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb
def basic_gp_regression_1d_tutorial():
	GPy.plotting.change_plotting_library('plotly')

	X = np.random.uniform(-3.0, 3.0, (20, 1))
	Y = np.sin(X) + np.random.randn(20, 1) * 0.05

	kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)
	model = GPy.models.GPRegression(X, Y, kernel)
	display(model)

	#fig = model.plot()
	#GPy.plotting.show(fig, filename='basic_gp_regression_notebook')

	model.optimize(messages=True)
	model.optimize_restarts(num_restarts=10)
	display(model)

	#fig = model.plot(plot_density=True)
	#GPy.plotting.show(fig, filename='basic_gp_regression_notebook_optimized')

# REF [site] >> http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_gp.ipynb
def basic_gp_regression_2d_tutorial():
	# Sample inputs and outputs.
	X = np.random.uniform(-3.0, 3.0, (50, 2))
	Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50, 1) * 0.05

	# Define kernel.
	kernel = GPy.kern.Matern52(2, ARD=True) + GPy.kern.White(2)

	# Create simple GP model.
	model = GPy.models.GPRegression(X, Y, kernel)

	# Optimize.
	model.optimize(messages=True, max_f_eval=1000)
	display(model)

	"""
	# Plot.
	fig = model.plot(plot_density=True)
	display(GPy.plotting.show(fig, filename='basic_gp_regression_notebook_2d'))

	# Plot slices.
	slices = [-1, 0, 1.5]
	figure = GPy.plotting.plotting_library().figure(
		3, 1, 
		shared_xaxes=True,
		subplot_titles=('slice at -1', 'slice at 0', 'slice at 1.5',)
	)
	for i, y in zip(range(3), slices):
		canvas = model.plot(figure=figure, fixed_inputs=[(1, y)], row=(i + 1), plot_data=False)
	GPy.plotting.show(canvas, filename='basic_gp_regression_notebook_slicing')
	"""

# REF [site] >> http://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/basic_kernels.ipynb
def basic_kernel_tutorial():
	%matplotlib inline
	%config InlineBackend.figure_format = 'svg'
	matplotlib.rcParams['figure.figsize'] = (8, 5)

	ker1 = GPy.kern.RBF(1)  # Equivalent to ker1 = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0).
	ker2 = GPy.kern.RBF(input_dim=1, variance=0.75, lengthscale=2.0)
	ker3 = GPy.kern.RBF(1, 0.5, 0.5)

	_ = ker1.plot(ax=plt.gca())
	_ = ker2.plot(ax=plt.gca())
	_ = ker3.plot(ax=plt.gca())

	# Implemented kernels.
	figure, axes = plt.subplots(3, 3, figsize=(10,10), tight_layout=True)
	kerns = [GPy.kern.RBF(1), GPy.kern.Exponential(1), GPy.kern.Matern32(1), GPy.kern.Matern52(1), GPy.kern.Brownian(1), GPy.kern.Bias(1), GPy.kern.Linear(1), GPy.kern.PeriodicExponential(1), GPy.kern.White(1)]
	for k, a in zip(kerns, axes.flatten()):
		k.plot(ax=a, x=1)
		a.set_title(k.name.replace('_', ' '))

	# Operations to combine kernels.
	# Product of kernels.
	k1 = GPy.kern.RBF(1, 1.0, 2.0)
	k2 = GPy.kern.Matern32(1, 0.5, 0.2)
	k_prod = k1 * k2
	print('Product of kernels =', k_prod)
	k_prod.plot()

	# Sum of kernels.
	k1 = GPy.kern.RBF(1, 1.0, 2.0)
	k2 = GPy.kern.Matern32(1, 0.5, 0.2)
	k_add = k1 + k2
	print('Sum of kernels =', k_add)
	k_add.plot()

	print('k1 =', k1)
	k_add.rbf.variance = 12.0
	print('k1 =', k1)

	# Operating on different domains
	k1 = GPy.kern.Linear(input_dim=1, active_dims=[0])  # Works on the first column of X, index=0.
	k2 = GPy.kern.ExpQuad(input_dim=1, lengthscale=3, active_dims=[1])  # Works on the second column of X, index=1.
	k = k1 * k2
	k.plot(x=np.ones((1, 2)))

def main():
	basic_model_tutorial()
	basic_gp_regression_1d_tutorial()
	basic_gp_regression_2d_tutorial()
	#basic_kernel_tutorial()

#%%------------------------------------------------------------------

# Usage:
#	python gpy_basic.py

if '__main__' == __name__:
	main()
