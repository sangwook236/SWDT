#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import sklearn.feature_selection, sklearn.datasets, sklearn.svm, sklearn.ensemble, sklearn.linear_model
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-learn.org/stable/modules/feature_selection.html
def basic_example():
	X, y =  sklearn.datasets.load_iris(return_X_y=True)
	print('Input shape (before) = {}.'.format(X.shape))

	clf = sklearn.svm.LinearSVC(C=0.01, penalty='l1', dual=False)
	clf = clf.fit(X, y)

	# The estimator should have a feature_importances_ or coef_ attribute after fitting. Otherwise, the importance_getter parameter should be used.
	model = sklearn.feature_selection.SelectFromModel(clf, prefit=True)
	X_new = model.transform(X)

	print('Input shape (after) = {}.'.format(X_new.shape))

	#--------------------
	clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=50)
	clf = clf.fit(X, y)
	print('Feature importance = {}.'.format(clf.feature_importances_))

	model = sklearn.feature_selection.SelectFromModel(clf, prefit=True)
	X_new = model.transform(X)

	print('Input shape (after) = {}.'.format(X_new.shape))

# REF [site] >> https://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_diabetes.html
def sequential_feature_selection_example():
	diabetes = sklearn.datasets.load_diabetes()
	X, y = diabetes.data, diabetes.target
	#print(diabetes.DESCR)

	# Feature importance from coefficients.
	ridge = sklearn.linear_model.RidgeCV(alphas=np.logspace(-6, 6, num=5))
	ridge = ridge.fit(X, y)

	importance = np.abs(ridge.coef_)
	feature_names = np.array(diabetes.feature_names)
	plt.bar(height=importance, x=feature_names)
	plt.title('Feature importances via coefficients')

	# Selecting features based on importance.
	threshold = np.sort(importance)[-3] + 0.01

	tic = time.time()
	sfm = sklearn.feature_selection.SelectFromModel(ridge, threshold=threshold)
	sfm = sfm.fit(X, y)
	toc = time.time()
	print(f'Features selected by SelectFromModel: {feature_names[sfm.get_support(indices=False)]}')
	print(f'Done in {toc - tic:.3f}s')
	#X_new = sfm.transform(X)

	# Selecting features with sequential feature selection.
	tic_fwd = time.time()
	sfs_forward = sklearn.feature_selection.SequentialFeatureSelector(ridge, n_features_to_select=2, direction='forward')
	#sfs_forward = sklearn.feature_selection.SequentialFeatureSelector(ridge, n_features_to_select='auto', tol=None, direction='forward', scoring=None, cv=None, n_jobs=None)
	sfs_forward = sfs_forward.fit(X, y)
	toc_fwd = time.time()
	#X_new = sfs_forward.transform(X)

	tic_bwd = time.time()
	sfs_backward = sklearn.feature_selection.SequentialFeatureSelector(ridge, n_features_to_select=2, direction='backward')
	#sfs_backward = sklearn.feature_selection.SequentialFeatureSelector(ridge, n_features_to_select='auto', tol=None, direction='backward', scoring=None, cv=None, n_jobs=None)
	sfs_backward = sfs_backward.fit(X, y)
	toc_bwd = time.time()
	#X_new = sfs_backward.transform(X)

	print(
		'Features selected by forward sequential selection: '
		f'{feature_names[sfs_forward.get_support(indices=False)]}'
	)
	print(f'Done in {toc_fwd - tic_fwd:.3f}s')
	print(
		'Features selected by backward sequential selection: '
		f'{feature_names[sfs_backward.get_support(indices=False)]}'
	)
	print(f'Done in {toc_bwd - tic_bwd:.3f}s')

	plt.show()

def main():
	#basic_example()
	sequential_feature_selection_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
