#!/usr/bin/env python

# REF [site] >> http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

def main():
	#X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
	#Y = np.array([1, 1, 2, 2])
	iris = load_iris()
	X = iris.data
	Y = iris.target
	#X, Y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)

	clf = LogisticRegression(penalty='l2')
	clf.fit(X, Y)

	#print('Prediction =', clf.predict([[-0.8, -1]]))
	#print('Prediction =', clf.predict([[5.1, 3.5, 1.4, 0.2]]))
	print('Prediction =', clf.predict(X))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
