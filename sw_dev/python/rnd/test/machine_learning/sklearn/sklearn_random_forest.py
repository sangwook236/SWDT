#!/usr/bin/env python

# REF [site] >> http://scikit-learn.org/stable/modules/ensemble.html
# REF [site] >> http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html

from sklearn import ensemble
from sklearn import datasets
import numpy as np

#%%-------------------------------------------------------------------

def random_forest_classifier_example():
	X, Y = datasets.make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, shuffle=False, random_state=0)
	#X, Y = datasets.make_blobs(n_samples=1000, n_features=4, centers=2, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=False, random_state=0)

	classifier = ensemble.RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=2, random_state=0)
	classifier.fit(X, Y)

	print('Feature importance =', classifier.feature_importances_)

	X_test = [[0, 0, 0, 0]]
	#X_test = X
	print('Prediction =', classifier.predict(X_test))
	print('Prediction (probability) =', classifier.predict_proba(X_test))
	print('Prediction (log probability) =', classifier.predict_log_proba(X_test))
	print('Score =', classifier.score(X, Y))

#%%-------------------------------------------------------------------

def random_forest_regressor_example():
	X, Y = datasets.make_regression(n_samples=1000, n_features=4, n_informative=2, n_targets=1, shuffle=False, random_state=0)

	regressor = ensemble.RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=2, random_state=0)
	regressor.fit(X, Y)

	print('Feature importances =', regressor.feature_importances_)

	X_test = [[0, 0, 0, 0]]
	#X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

def main():
	random_forest_classifier_example()
	random_forest_regressor_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
