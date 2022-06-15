#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import sklearn.gaussian_process, sklearn.kernel_ridge, sklearn.model_selection, sklearn.utils, sklearn.datasets
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-learn.org/stable/modules/gaussian_process.html
def basic_operation():
	# Kernels are parameterized by a vector 'theta' of hyperparameters
	# These hyperparameters can for instance control length-scales or periodicity of a kernel.
	# All kernels support computing analytic gradients of the kernel's auto-covariance with respect to log(theta) via setting eval_gradient=True in the __call__ method.
	# This gradient is used by the Gaussian process (both regressor and classifier) in computing the gradient of the log-marginal-likelihood, which in turn is used to determine the value of 'theta', which maximizes the log-marginal-likelihood, via gradient ascent.
	# For each hyperparameter, the initial value and the bounds need to be specified when creating an instance of the kernel.
	# The current value of 'theta' can be get and set via the property 'theta' of the kernel object.
	# Moreover, the bounds of the hyperparameters can be accessed by the property 'bounds' of the kernel.
	# Note that both properties (theta and bounds) return log-transformed values of the internally used values since those are typically more amenable to gradient-based optimization.
	# The specification of each hyperparameter is stored in the form of an instance of Hyperparameter in the respective kernel.
	# Note that a kernel using a hyperparameter with name "x" must have the attributes self.x and self.x_bounds.

	# Note that due to the nested structure of kernels, the names of kernel parameters might become relatively complicated.
	# In general, for a binary kernel operator, parameters of the left operand are prefixed with k1__ and parameters of the right operand with k2__.

	# The main usage of a Kernel is to compute the GP's covariance between datapoints.
	kernel = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
		* sklearn.gaussian_process.kernels.RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) \
		+ sklearn.gaussian_process.kernels.RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))

	print("Hyperparameters:")
	for hyperparameter in kernel.hyperparameters:
		print("\t{}.".format(hyperparameter))

	print("Parameters:")
	params = kernel.get_params()
	for key in sorted(params):
		print("\t{} : {}.".format(key, params[key]))

	print(kernel.theta)  # Note: log-transformed.
	print(kernel.bounds)  # Note: log-transformed.

# REF [site] >> https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html
def gpr_example():
	rng = np.random.RandomState(0)
	data = np.linspace(0, 30, num=1_000).reshape(-1, 1)
	target = np.sin(data).ravel()

	training_sample_indices = rng.choice(np.arange(0, 400), size=40, replace=False)
	training_data = data[training_sample_indices]
	training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(len(training_sample_indices))

	plt.figure()
	plt.plot(data, target, label="True signal", linewidth=2)
	plt.scatter(
		training_data,
		training_noisy_target,
		color="black",
		label="Noisy measurements",
	)
	plt.legend()
	plt.xlabel("data")
	plt.ylabel("target")
	_ = plt.title(
		"Illustration of the true generative process and \n"
		"noisy measurements available during training"
	)

	# REF [site] >> https://scikit-learn.org/stable/modules/gaussian_process.html
	# Comparison of GPR and Kernel Ridge Regression.
	#	A major difference is that GPR can choose the kernel's hyperparameters based on gradient-ascent on the marginal likelihood function while KRR needs to perform a grid search on a cross-validated loss function (mean-squared error loss).
	#	A further difference is that GPR learns a generative, probabilistic model of the target function and can thus provide meaningful confidence intervals and posterior samples along with the predictions while KRR only provides predictions.

	#--------------------
	# Kernel ridge regression.

	kernel_ridge = sklearn.kernel_ridge.KernelRidge(kernel=sklearn.gaussian_process.kernels.ExpSineSquared())

	start_time = time.time()
	kernel_ridge.fit(training_data, training_noisy_target)
	print(f"Fitting KernelRidge with default kernel: {time.time() - start_time:.3f} seconds")

	plt.figure()
	plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
	plt.scatter(
		training_data,
		training_noisy_target,
		color="black",
		label="Noisy measurements",
	)
	plt.plot(
		data,
		kernel_ridge.predict(data),
		label="Kernel ridge",
		linewidth=2,
		linestyle="dashdot",
	)
	plt.legend(loc="lower right")
	plt.xlabel("data")
	plt.ylabel("target")
	_ = plt.title(
		"Kernel ridge regression with an exponential sine squared\n "
		"kernel using default hyperparameters"
	)

	print('Kernel = {}.'.format(kernel_ridge.kernel))

	# The kernel's hyperparameters control the smoothness (length_scale) and periodicity of the kernel (periodicity).
	# The noise level of the data is learned explicitly by the regularization parameter alpha of KRR.
	param_distributions = {
		"alpha": sklearn.utils.fixes.loguniform(1e0, 1e3),
		"kernel__length_scale": sklearn.utils.fixes.loguniform(1e-2, 1e2),
		"kernel__periodicity": sklearn.utils.fixes.loguniform(1e0, 1e1),
	}
	kernel_ridge_tuned = sklearn.model_selection.RandomizedSearchCV(
		kernel_ridge,
		param_distributions=param_distributions,
		n_iter=500,
		random_state=0,
	)
	start_time = time.time()
	kernel_ridge_tuned.fit(training_data, training_noisy_target)
	print(f"Time for KernelRidge fitting: {time.time() - start_time:.3f} seconds")

	print('Best params = {}.'.format(kernel_ridge_tuned.best_params_))

	start_time = time.time()
	predictions_kr = kernel_ridge_tuned.predict(data)
	print(f"Time for KernelRidge predict: {time.time() - start_time:.3f} seconds")

	plt.figure()
	plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
	plt.scatter(
		training_data,
		training_noisy_target,
		color="black",
		label="Noisy measurements",
	)
	plt.plot(
		data,
		predictions_kr,
		label="Kernel ridge",
		linewidth=2,
		linestyle="dashdot",
	)
	plt.legend(loc="lower right")
	plt.xlabel("data")
	plt.ylabel("target")
	_ = plt.title(
		"Kernel ridge regression with an exponential sine squared\n "
		"kernel using tuned hyperparameters"
	)

	#--------------------
	# Gaussian process regression.

	# The kernel's hyperparameters control the smoothness (length_scale) and periodicity of the kernel (periodicity).
	# The noise level of the data is learned explicitly by GPR by an additional WhiteKernel component in the kernel.
	kernel = 1.0 * sklearn.gaussian_process.kernels.ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
		+ sklearn.gaussian_process.kernels.WhiteKernel(1e-1)
	gaussian_process = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
	start_time = time.time()
	gaussian_process.fit(training_data, training_noisy_target)
	print(f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds")

	print('Kernel = {}.'.format(gaussian_process.kernel))

	start_time = time.time()
	mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(data, return_std=True)
	print(f"Time for GaussianProcessRegressor predict: {time.time() - start_time:.3f} seconds")

	plt.figure()
	plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
	plt.scatter(
		training_data,
		training_noisy_target,
		color="black",
		label="Noisy measurements",
	)
	# Plot the predictions of the kernel ridge.
	plt.plot(
		data,
		predictions_kr,
		label="Kernel ridge",
		linewidth=2,
		linestyle="dashdot",
	)
	# Plot the predictions of the gaussian process regressor.
	plt.plot(
		data,
		mean_predictions_gpr,
		label="Gaussian process regressor",
		linewidth=2,
		linestyle="dotted",
	)
	plt.fill_between(
		data.ravel(),
		mean_predictions_gpr - std_predictions_gpr,
		mean_predictions_gpr + std_predictions_gpr,
		color="tab:green",
		alpha=0.2,
	)
	plt.legend(loc="lower right")
	plt.xlabel("data")
	plt.ylabel("target")
	_ = plt.title("Comparison between kernel ridge and gaussian process regressor")

	# The kernel's hyperparameters control the smoothness (length_scale) and periodicity of the kernel (periodicity).
	# The noise level of the data is learned explicitly by GPR by an additional WhiteKernel component in the kernel.
	kernel = 1.0 * sklearn.gaussian_process.kernels.ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
		* sklearn.gaussian_process.kernels.RBF(length_scale=15, length_scale_bounds="fixed") \
		+ sklearn.gaussian_process.kernels.WhiteKernel(1e-1)
	gaussian_process = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel)
	gaussian_process.fit(training_data, training_noisy_target)
	mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(data, return_std=True)

	plt.figure()
	plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
	plt.scatter(
		training_data,
		training_noisy_target,
		color="black",
		label="Noisy measurements",
	)
	# Plot the predictions of the kernel ridge.
	plt.plot(
		data,
		predictions_kr,
		label="Kernel ridge",
		linewidth=2,
		linestyle="dashdot",
	)
	# Plot the predictions of the gaussian process regressor.
	plt.plot(
		data,
		mean_predictions_gpr,
		label="Gaussian process regressor",
		linewidth=2,
		linestyle="dotted",
	)
	plt.fill_between(
		data.ravel(),
		mean_predictions_gpr - std_predictions_gpr,
		mean_predictions_gpr + std_predictions_gpr,
		color="tab:green",
		alpha=0.2,
	)
	plt.legend(loc="lower right")
	plt.xlabel("data")
	plt.ylabel("target")
	_ = plt.title("Effect of using a radial basis function kernel")

	plt.show()

# REF [site] >> https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpc_iris.html
def gpc_example():
	iris = sklearn.datasets.load_iris()
	X = iris.data[:, :2]  # We only take the first two features.
	y = np.array(iris.target, dtype=int)

	h = 0.02  # Step size in the mesh.

	kernel = 1.0 * sklearn.gaussian_process.kernels.RBF([1.0])
	gpc_rbf_isotropic = sklearn.gaussian_process.GaussianProcessClassifier(kernel=kernel).fit(X, y)
	kernel = 1.0 * sklearn.gaussian_process.kernels.RBF([1.0, 1.0])
	gpc_rbf_anisotropic = sklearn.gaussian_process.GaussianProcessClassifier(kernel=kernel).fit(X, y)

	# Create a mesh to plot in.
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	titles = ["Isotropic RBF", "Anisotropic RBF"]
	plt.figure(figsize=(10, 5))
	for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
		# Plot the predicted probabilities. For that, we will assign a color to
		# each point in the mesh [x_min, m_max]x[y_min, y_max].
		plt.subplot(1, 2, i + 1)

		Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

		# Put the result into a color plot.
		Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
		plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

		# Plot also the training points.
		plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
		plt.xlabel("Sepal length")
		plt.ylabel("Sepal width")
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.xticks(())
		plt.yticks(())
		plt.title("%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta)))
	plt.tight_layout()

	plt.show()

def main():
	#basic_operation()

	# The GaussianProcessClassifier implements Gaussian processes (GP) for classification purposes, more specifically for probabilistic classification, where test predictions take the form of class probabilities.
	# GaussianProcessClassifier places a GP prior on a latent function f, which is then squashed through a link function to obtain the probabilistic classification.
	# The latent function f is a so-called nuisance function, whose values are not observed and are not relevant by themselves.
	# Its purpose is to allow a convenient formulation of the model, and f is removed (integrated out) during prediction.
	# GaussianProcessClassifier implements the logistic link function, for which the integral cannot be computed analytically but is easily approximated in the binary case.
	# In contrast to the regression setting, the posterior of the latent function f is not Gaussian even for a GP prior since a Gaussian likelihood is inappropriate for discrete class labels.
	# Rather, a non-Gaussian likelihood corresponding to the logistic link function (logit) is used.
	# GaussianProcessClassifier approximates the non-Gaussian posterior with a Gaussian based on the Laplace approximation.

	gpr_example()
	#gpc_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
