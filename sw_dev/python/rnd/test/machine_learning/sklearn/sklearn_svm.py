#!/usr/bin/env python

# REF [site] >> http://scikit-learn.org/stable/modules/ensemble.html
# REF [site] >> http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.svm import libsvm
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
import numpy as np

#%%-------------------------------------------------------------------

def svc_example():
	#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	#Y = np.array([1, 1, 2, 2])
	iris = load_iris()
	X = iris.data
	Y = iris.target
	#X, Y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	clf = SVC()
	clf.fit(X, Y) 

	#print('Prediction =', clf.predict([[-0.8, -1]]))
	#print('Prediction =', clf.predict([[5.1, 3.5, 1.4, 0.2]]))
	print('Prediction =', clf.predict(X))

#%%-------------------------------------------------------------------
# Linear SVC.

def linear_svc_example():
	iris = load_iris()
	X = iris.data
	Y = iris.target

	clf = LinearSVC(penalty='l2', loss='squared_hinge')
	clf.fit(X, Y) 

	print('Prediction =', clf.predict(X))

#%%-------------------------------------------------------------------

def svr_example():
	num_samples, num_features = 10, 5
	np.random.seed(0)
	Y = np.random.randn(num_samples)
	X = np.random.randn(num_samples, num_features)

	clf = SVR(C=1.0, epsilon=0.2)
	clf.fit(X, Y) 

	#print('Prediction =', clf.predict(np.random.randn(1, num_features)))
	print('Prediction =', clf.predict(X))

#%%-------------------------------------------------------------------
# libsvm.

def libsvmc_example():
	iris = load_iris()
	X = iris.data.astype(float)
	Y = iris.target.astype(float)
	#X, Y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	support, SV, n_class_SV, sv_coef, intercept, probA, probB, fit_status = libsvm.fit(X, Y, svm_type=0, kernel='rbf', degree=3, gamma=0.1, coef0=0, tol=1e-3, C=1.0, nu=0.5, epsilon=0.1, class_weight=np.empty(0), sample_weight=np.empty(0), shrinking=1, probability=1, cache_size=100.0, max_iter=-1, random_seed=0) 
	#print(probA, probB)

	pred1 = libsvm.predict(X[50,:].reshape((1, -1)), support, SV, n_class_SV, sv_coef, intercept, probA, probB, svm_type=0, kernel='rbf', degree=3, gamma=0.1, coef0=0, class_weight=np.empty(0), sample_weight=np.empty(0), cache_size=100.0)
	pred2 = libsvm.predict_proba(X[50,:].reshape((1, -1)), support, SV, n_class_SV, sv_coef, intercept, probA, probB, svm_type=0, kernel='rbf', degree=3, gamma=0.1, coef0=0, class_weight=np.empty(0), sample_weight=np.empty(0), cache_size=100.0)
	print('Prediction =', pred1, pred2)

def main():
	svc_example()
	linear_svc_example()
	svr_example()
	libsvmc_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
