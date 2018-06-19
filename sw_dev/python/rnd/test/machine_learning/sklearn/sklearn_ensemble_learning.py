#!/usr/bin/env python

# REF [site] >> http://scikit-learn.org/stable/modules/ensemble.html
# REF [site] >> http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

#%%-------------------------------------------------------------------

def main():
	iris = load_iris()
	X = iris.data
	y = iris.target
	#X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	num_estimators = 30

	clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
	scores = cross_val_score(clf, X, y)
	print("DecisionTreeClassifier =", scores.mean())

	clf = RandomForestClassifier(n_estimators=num_estimators, max_depth=None, min_samples_split=2, random_state=0)
	scores = cross_val_score(clf, X, y)
	print("RandomForestClassifier =", scores.mean())

	clf = ExtraTreesClassifier(n_estimators=num_estimators, max_depth=None, min_samples_split=2, random_state=0)
	scores = cross_val_score(clf, X, y)
	print("ExtraTreesClassifier =", scores.mean())

	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=num_estimators)
	scores = cross_val_score(clf, X, y)
	print("AdaBoostClassifier =", scores.mean())

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
