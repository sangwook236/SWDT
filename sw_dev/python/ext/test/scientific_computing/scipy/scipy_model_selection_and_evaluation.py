#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.model_selection, sklearn.datasets, sklearn.svm, sklearn.linear_model, sklearn.utils
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-learn.org/stable/modules/cross_validation.html
def cross_validation_example():
	raise NotImplementedError

# REF [site] >> https://scikit-learn.org/stable/modules/grid_search.html
def hyper_parameter_tuning_example():
	# Hyper-parameters are parameters that are not directly learnt within estimators.
	# In scikit-learn they are passed as arguments to the constructor of the estimator classes.
	# Typical examples include C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc.
	# It is possible and recommended to search the hyper-parameter space for the best cross validation score.
	# Any parameter provided when constructing an estimator may be optimized in this manner.
	# Specifically, to find the names and current values for all parameters for a given estimator:
	#	estimator.get_params()

	"""
	parameters = {
		'C': scipy.stats.expon(scale=100),
		'gamma': scipy.stats.expon(scale=0.1),
		'kernel': ['rbf'],
		'class_weight':['balanced', None]}
	parameters = {
		'C': sklearn.utils.fixes.loguniform(1e0, 1e3),
		'gamma': sklearn.utils.fixes.loguniform(1e-4, 1e-3),
		'kernel': ['rbf'],
		'class_weight':['balanced', None]
	}
	"""

	#--------------------
	# Exhaustive grid search.
	# REF [site] >> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

	if True:
		iris = sklearn.datasets.load_iris()

		#param_grid = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
		param_grid = [
			{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
			{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
		]
		clf = sklearn.svm.SVC()
		search = sklearn.model_selection.GridSearchCV(clf, param_grid)
		search.fit(iris.data, iris.target)

		print('CV keys = {}.'.format(sorted(search.cv_results_.keys())))
		print(pd.DataFrame(search.cv_results_))
		print('Best params: {}.'.format(search.best_params_))
		print('Best estimator: {}.'.format(search.best_estimator_))
		print('Best score = {}.'.format(search.best_score_))

	#--------------------
	# Randomized search.
	# REF [site] >> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

	if True:
		iris = sklearn.datasets.load_iris()

		clf = sklearn.linear_model.LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)
		param_distributions ={
			'C': scipy.stats.uniform(loc=0, scale=4),
			'penalty': ['l2', 'l1']
		}
		search = sklearn.model_selection.RandomizedSearchCV(clf, param_distributions, random_state=0)
		search = search.fit(iris.data, iris.target)

		print('CV keys = {}.'.format(sorted(search.cv_results_.keys())))
		print(pd.DataFrame(search.cv_results_))
		print('Best params: {}.'.format(search.best_params_))
		print('Best estimator: {}.'.format(search.best_estimator_))
		print('Best score = {}.'.format(search.best_score_))

	#--------------------
	# Randomized parameter optimization.
	# REF [site] >> https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

	if True:
		X, y = sklearn.datasets.load_digits(return_X_y=True, n_class=3)

		# Build a classifier.
		clf = sklearn.linear_model.SGDClassifier(loss="hinge", penalty="elasticnet", fit_intercept=True)

		# Utility function to report best scores.
		def report(results, n_top=3):
			for i in range(1, n_top + 1):
				candidates = np.flatnonzero(results["rank_test_score"] == i)
				for candidate in candidates:
					print("Model with rank: {0}".format(i))
					print("Mean validation score: {0:.3f} (std: {1:.3f})".format(results["mean_test_score"][candidate], results["std_test_score"][candidate]))
					print("Parameters: {0}".format(results["params"][candidate]))
					print("")

		# Specify parameters and distributions to sample from.
		param_dist = {
			"average": [True, False],
			"l1_ratio": scipy.stats.uniform(0, 1),
			"alpha": sklearn.utils.fixes.loguniform(1e-2, 1e0),
		}

		# Run randomized search.
		n_iter_search = 15
		random_search = sklearn.model_selection.RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

		start = time.time()
		random_search.fit(X, y)
		print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time.time() - start), n_iter_search) )
		report(random_search.cv_results_)

		# Use a full grid over all parameters.
		param_grid = {
			"average": [True, False],
			"l1_ratio": np.linspace(0, 1, num=10),
			"alpha": np.power(10, np.arange(-2, 1, dtype=float)),
		}

		# Run grid search.
		grid_search = sklearn.model_selection.GridSearchCV(clf, param_grid=param_grid)
		start = time.time()
		grid_search.fit(X, y)

		print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time.time() - start, len(grid_search.cv_results_["params"])))
		report(grid_search.cv_results_)

	#--------------------
	# Searching for optimal parameters with successive halving.
	# REF [site] >> https://scikit-learn.org/stable/modules/grid_search.html

# REF [site] >> https://scikit-learn.org/stable/modules/model_evaluation.html
def model_evaluation_example():
	raise NotImplementedError

# REF [site] >> https://scikit-learn.org/stable/modules/learning_curve.html
def learning_curve_example():
	raise NotImplementedError

def main():
	#cross_validation_example()  # Not yet implemented.
	hyper_parameter_tuning_example()
	#model_evaluation_example()  # Not yet implemented.
	#learning_curve_example()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
