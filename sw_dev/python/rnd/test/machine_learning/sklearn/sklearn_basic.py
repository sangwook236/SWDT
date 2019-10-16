import numpy as np
from sklearn.externals import joblib

#---------------------------------------------------------------------
# Dataset split.
# REF [site] >> http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

from sklearn.model_selection import train_test_split

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
print(X_train, X_test, y_train, y_test)

#---------------------------------------------------------------------
# k-fold.
# REF [site] >> http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-i-i-d-data

from sklearn.model_selection import KFold

kfold = KFold(n_splits=2)
print('#splits =', kfold.get_n_splits(X))

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])

for train_index, test_index in kfold.split(X):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
