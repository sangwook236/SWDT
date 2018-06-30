#!/usr/bin/env python

from sklearn import multiclass, multioutput
from sklearn import svm, ensemble
from sklearn import datasets, preprocessing, utils
import numpy as np

# REF [site] >>
#	http://scikit-learn.org/stable/modules/multiclass.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html
#	http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html

def multiclass_example():
	iris = datasets.load_iris()
	X, y = iris.data, iris.target

	random_state = 0
	clf = svm.LinearSVC(random_state=random_state)

	#--------------------
	ovr_clf = multiclass.OneVsRestClassifier(clf)
	ovr_clf.fit(X, y)
	pred = ovr_clf.predict(X)
	print('Prediction (ovr) =\n', pred)

	#--------------------
	ovo_clf = multiclass.OneVsOneClassifier(clf)
	ovo_clf.fit(X, y)
	pred = ovo_clf.predict(X)
	print('Prediction (ovo) =\n', pred)

	#--------------------
	oc_clf = multiclass.OutputCodeClassifier(clf, code_size=2, random_state=random_state)
	oc_clf.fit(X, y)
	pred = oc_clf.predict(X)
	print('Prediction (oc) =\n', pred)

def multioutput_classification_example():
	X, y1 = datasets.make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
	y2 = utils.shuffle(y1, random_state=1)
	y3 = utils.shuffle(y1, random_state=2)
	Y = np.vstack((y1, y2, y3)).T

	n_samples, n_features = X.shape  # 10, 100.
	n_outputs = Y.shape[1]  # 3.
	n_classes = 3
	clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=1)

	mo_clf = multioutput.MultiOutputClassifier(clf, n_jobs=-1)
	mo_clf.fit(X, Y)
	pred = mo_clf.predict(X)
	print('Prediction =\n', pred)

def multioutput_regression_example():
	X, y = datasets.make_regression(n_samples=10, n_targets=3, random_state=1)

	regr = ensemble.GradientBoostingRegressor(random_state=0)

	mo_regr = multioutput.MultiOutputRegressor(regr)
	mo_regr.fit(X, y)
	pred = mo_regr.predict(X)
	print('Prediction =\n', pred)

def main():
	y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
	y_transformed = preprocessing.MultiLabelBinarizer().fit_transform(y)
	print('y_transformed =\n', y_transformed)

	multiclass_example()

	multioutput_classification_example()
	multioutput_regression_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
