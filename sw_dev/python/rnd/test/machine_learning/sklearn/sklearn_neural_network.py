#!/usr/bin/env python

# REF [site] >> http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn import neural_network
from sklearn import datasets
import numpy as np

def mlp_classification_example():
	iris = datasets.load_iris()
	X, Y = iris.data, iris.target
	#X, Y = datasets.make_classification(n_samples=10000, n_features=10, n_informative=2, n_redundant=0, n_repeated=0, n_classes=100, shuffle=False, random_state=0)

	classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	classifier.fit(X, Y)

	for id, coef in enumerate(classifier.coefs_):
		print('{}-th weights {} = \n{}'.format(id, coef.shape, coef))

	print('Prediction =', classifier.predict(X))

def mlp_regression_example():
	num_samples, num_features = 10, 5
	np.random.seed(0)
	X, Y = np.random.randn(num_samples, num_features), np.random.randn(num_samples)
	#X, Y = datasets.make_regression(n_samples=num_samples, n_features=num_features, n_informative=2, n_targets=1, shuffle=False, random_state=0)

	regressor = neural_network.MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	regressor.fit(X, Y)

	for (id, coef) in enumerate(regressor.coefs_):
		print('{}-th weights {} = \n{}'.format(id, coef.shape, coef))

	X_test = np.random.randn(1, num_features)
	#X_test = X
	print('Prediction =', regressor.predict(X_test))
	print('Score =', regressor.score(X, Y))

def main():
	mlp_classification_example()
	mlp_regression_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
