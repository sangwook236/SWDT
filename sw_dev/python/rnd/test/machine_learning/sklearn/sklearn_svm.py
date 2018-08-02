#!/usr/bin/env python

# REF [site] >>
#	http://scikit-learn.org/stable/modules/svm.html
#	http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
#	http://scikit-learn.org/stable/auto_examples/applications/svm_gui.html

from sklearn import svm
from sklearn import datasets
import numpy as np

#%%-------------------------------------------------------------------
# C-support vector classification.

def svc_example():
	#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	#Y = np.array([1, 1, 2, 2])
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target
	#X, Y = datasets.make_classification(n_samples=10000, n_features=10, n_informative=2, n_redundant=0, n_repeated=0, n_classes=100, shuffle=False, random_state=0)
	#X, Y = datasets.make_blobs(n_samples=10000, n_features=10, centers=100, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=False, random_state=0)

	probability = True
	classifier = svm.SVC(kernel='rbf', degree=3, probability=probability, decision_function_shape='ovr', random_state=None)
	#classifier = svm.SVC(kernel='linear', degree=3, probability=probability, decision_function_shape='ovr', random_state=None)
	classifier.fit(X, Y)

	if hasattr(classifier, 'coef_'):
		print('Coefficient =', classifier.coef_)  # Shape = ((num_classes-1) * num_classes / 2, num_features).
		print('Intercept =', classifier.intercept_)

		# Feature importance. (?)
		#	https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d.
		coef = classifier.coef_.ravel()  # coef_ is only available when using a linear kernel.
		#num_top_features = 20
		#top_positive_coefficients = np.argsort(coef)[-num_top_features:]
		#top_negative_coefficients = np.argsort(coef)[:num_top_features]
		#top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
		top_coefficients = np.argsort(coef)
		print('Top coefficients =', top_coefficients)
		#print('Top features =', feature_names[top_coefficients])

	#X_test = [[-0.8, -1]]
	#X_test = [[5.1, 3.5, 1.4, 0.2]]
	X_test = X
	print('Prediction =', classifier.predict(X_test))
	if probability:
		print('Prediction (probability) =', classifier.predict_proba(X_test))
		print('Prediction (log probability) =', classifier.predict_log_proba(X_test))
	# Signed distance to the separating hyperplane.
	#	Shape = (n_samples, n_classes) when decision_function_shape = 'ovr'.
	#	Shape = (n_samples, n_classes * (n_classes-1) / 2) when decision_function_shape = 'ovo'.
	print('Distance =', classifier.decision_function(X_test))
	print('Score =', classifier.score(X, Y))

#%%-------------------------------------------------------------------
# Linear SVC.

def linear_svc_example():
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target

	classifier = svm.LinearSVC(penalty='l2', loss='squared_hinge', multi_class='ovr', max_iter=1000, random_state=None)
	classifier.fit(X, Y)

	#classifier.densify()  # Converts coefficient matrix of dense array format.
	#classifier.sparsify()  # Converts coefficient matrix of sparse format.

	print('Coefficient =', classifier.coef_)  # Shape = ((num_classes-1) * num_classes / 2, num_features).
	print('Intercept =', classifier.intercept_)

	# Feature importance. (?)
	coef = classifier.coef_.ravel()
	top_coefficients = np.argsort(coef)
	print('Top coefficients =', top_coefficients)

	X_test = X
	print('Prediction =', classifier.predict(X_test))
	# Signed distance to the separating hyperplane.
	#	Shape = (n_samples,) if n_classes == 2.
	#	Shape = (n_samples, n_classes) if n_classes > 2.
	print('Distance =', classifier.decision_function(X_test))
	print('Score =', classifier.score(X, Y))

#%%-------------------------------------------------------------------
# Nu SVC: Nu-support vector classification.

def nu_svc_example():
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target

	probability = True
	classifier = svm.NuSVC(nu=0.5, kernel='rbf', degree=3, probability=probability, max_iter=-1, decision_function_shape='ovr', random_state=None)
	classifier.fit(X, Y)

	X_test = X
	print('Prediction =', classifier.predict(X_test))
	if probability:
		print('Prediction (probability) =', classifier.predict_proba(X_test))
		print('Prediction (log probability) =', classifier.predict_log_proba(X_test))
	# Signed distance to the separating hyperplane.
	#	Shape = (n_samples, n_classes) when decision_function_shape = 'ovr'.
	#	Shape = (n_samples, n_classes * (n_classes-1) / 2) when decision_function_shape = 'ovo'.
	print('Distance =', classifier.decision_function(X_test))
	print('Score =', classifier.score(X, Y))

#%%-------------------------------------------------------------------
# Epsilon-support vector regression.

def svr_example():
	num_samples, num_features = 10, 5
	np.random.seed(0)
	X, Y = np.random.randn(num_samples, num_features), np.random.randn(num_samples)
	#X, Y = datasets.make_regression(n_samples=num_samples, n_features=num_features, n_informative=2, n_targets=1, shuffle=False, random_state=0)

	regressor = svm.SVR(epsilon=0.1, C=1.0, kernel='rbf', degree=3, max_iter=-1)
	regressor.fit(X, Y)

	X_test = np.random.randn(1, num_features)
	#X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

#%%-------------------------------------------------------------------
# Linear SVR.

def linear_svr_example():
	#X, y = datasets.make_regression(n_features=4, random_state=0)
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target

	regressor = svm.LinearSVR(epsilon=0.0, C=1.0, loss='epsilon_insensitive', max_iter=1000, random_state=0)
	regressor.fit(X, Y)

	print('Coefficient =', regressor.coef_)
	print('Intercept =', regressor.intercept_)

	#X_test = [[0, 0, 0, 0]]
	X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

#%%-------------------------------------------------------------------
# Nu SVR: Nu-support vector regression.

def nu_svr_example():
	n_samples, n_features = 10, 5
	np.random.seed(0)
	X, Y = np.random.randn(n_samples, n_features), np.random.randn(n_samples)
	#iris = datasets.load_iris()
	#X, Y = iris.data, iris.target

	regressor = svm.NuSVR(nu=0.5, kernel='rbf', degree=3, max_iter=-1)
	regressor.fit(X, Y)

	X_test = np.random.randn(5, n_features)
	#X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

#%%-------------------------------------------------------------------
# One-class SVM.

def one_class_svm_example():
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target

	classifier = svm.OneClassSVM(kernel='rbf', degree=3, nu=0.5, max_iter=-1, random_state=0)
	classifier.fit(X)

	X_test = X
	print('Prediction =', classifier.predict(X_test))
	# Signed distance to the separating hyperplane: shape = (n_samples,).
	print('Distance =', classifier.decision_function(X_test))

#%%-------------------------------------------------------------------
# libsvm.

def libsvm_example():
	iris = datasets.load_iris()
	X = iris.data.astype(float)
	Y = iris.target.astype(float)
	#X, Y = datasets.make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	svm_type = 0
	kernel = 'rbf'
	degree = 3
	gamma = 0.1
	coef0 = 0
	C = 1.0
	nu = 0.5
	epsilon = 0.1
	shrinking = 1
	probability = 1

	support, SV, n_class_SV, sv_coef, intercept, probA, probB, fit_status = svm.libsvm.fit(X, Y, svm_type=svm_type, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, nu=nu, epsilon=epsilon, shrinking=shrinking, probability=probability, max_iter=-1, random_seed=0)
	#print(probA, probB)

	X_test = X[50,:].reshape((1, -1))
	pred = svm.libsvm.predict(X_test, support, SV, n_class_SV, sv_coef, intercept, probA, probB, svm_type=svm_type, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
	print('Prediction =', pred)
	pred_prob = svm.libsvm.predict_proba(X_test, support, SV, n_class_SV, sv_coef, intercept, probA, probB, svm_type=svm_type, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
	print('Prediction (probability) =', pred_prob)
	margin = svm.libsvm.decision_function(X_test, support, SV, n_class_SV, sv_coef, intercept, probA, probB, svm_type=svm_type, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0)
	print('Predicted margin =', margin)

	n_fold = 5
	cross_val = svm.libsvm.cross_validation(X, Y, n_fold, svm_type=svm_type, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, C=C, nu=nu, epsilon=epsilon, shrinking=shrinking, probability=probability, max_iter=-1, random_seed=0)
	print('Cross validation =', cross_val)

def main():
	svc_example()
	#linear_svc_example()
	#nu_svc_example()

	#svr_example()
	#linear_svr_example()
	#nu_svr_example()

	#one_class_svm_example()

	#libsvm_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
