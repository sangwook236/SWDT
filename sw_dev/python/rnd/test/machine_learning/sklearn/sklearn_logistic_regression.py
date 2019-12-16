#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://scikit-learn.org/stable/modules/linear_model.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn import linear_model
from sklearn import datasets
import numpy as np

def main():
	#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	#Y = np.array([1, 1, 2, 2])
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target
	#X, Y = datasets.make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	classifier = linear_model.LogisticRegression(penalty='l2')
	classifier.fit(X, Y)

	print('Coefficient =', classifier.coef_)  # Shape = (num_classes, num_features).
	print('Intercept =', classifier.intercept_)

	# Feature importance. (?)
	coef = classifier.coef_.ravel()
	top_coefficients = np.argsort(coef)
	print('Top coefficients =', top_coefficients)

	#print('Prediction =', classifier.predict([[-0.8, -1]]))
	#print('Prediction =', classifier.predict([[5.1, 3.5, 1.4, 0.2]]))
	print('Prediction =', classifier.predict(X))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
