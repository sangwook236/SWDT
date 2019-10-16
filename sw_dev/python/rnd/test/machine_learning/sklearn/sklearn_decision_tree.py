#!/usr/bin/env python

from sklearn import datasets
from sklearn import model_selection
from sklearn import tree

#---------------------------------------------------------------------

def decision_tree_classifier_example():
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target

	classifier = tree.DecisionTreeClassifier(random_state=0)
	classifier.fit(X, Y)

	print('Feature importance =', classifier.feature_importances_)

	X_test = [[0, 0, 0, 0]]
	#X_test = X
	print('Prediction =', classifier.predict(X_test))
	print('Prediction (probability) =', classifier.predict_proba(X_test))
	print('Prediction (log probability) =', classifier.predict_log_proba(X_test))
	print('Score =', classifier.score(X, Y))

	print('Decision path =', classifier.decision_path(X))
	print('Index of the leaf =', classifier.apply(X))

#---------------------------------------------------------------------

def decision_tree_regressor_example():
	boston = datasets.load_boston()
	X, Y = boston.data, boston.target

	regressor = tree.DecisionTreeRegressor(random_state=0)
	regressor.fit(X, Y)

	print('Feature importance =', regressor.feature_importances_)

	X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

	print('Decision path =', regressor.decision_path(X))
	print('Index of the leaf =', regressor.apply(X))

def main():
	decision_tree_classifier_example()
	decision_tree_regressor_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
