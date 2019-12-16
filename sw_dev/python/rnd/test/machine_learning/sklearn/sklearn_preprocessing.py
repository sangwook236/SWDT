#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> http://scikit-learn.org/stable/modules/preprocessing.html

from sklearn import preprocessing
import numpy as np

# Standardization, or mean removal and variance scaling.
def standardization_example():
	X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

	X_scaled = preprocessing.scale(X, axis=0, with_mean=True, with_std=True, copy=True)
	#X_scaled = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0, copy=True)  # [0, 1].
	#X_scaled = preprocessing.maxabs_scale(X, axis=0, copy=True)  # [-1, 1].
	#X_scaled = preprocessing.robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
	X_scaled.mean(axis=0)
	X_scaled.std(axis=0)

	scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
	#scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True).fit(X)  # [0, 1].
	#scaler = preprocessing.MaxAbsScaler(copy=True).fit(X)  # [-1, 1].
	#scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True).fit(X)
	print(scaler.mean_)
	print(scaler.scale_)
	scaler.transform(X)
	scaler.transform([[-1.0, 1.0, 0.0]])

# Normalization.
def normalization_example():
	X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

	X_normalized = preprocessing.normalize(X, norm='l2')

	normalizer = preprocessing.Normalizer().fit(X)
	normalizer.transform(X)
	#normalizer = preprocessing.Normalizer().fit_transform(X)

# Whitening.
def whitening_example():
	#sklearn.decomposition.PCA
	#sklearn.decomposition.RandomizedPCA

# Binarization.
def binarization_example():
	X = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])

	binarizer = preprocessing.Binarizer().fit(X)
	#binarizer = preprocessing.Binarizer(threshold=1.1)
	binarizer.transform(X)

# Encoding categorical features.
def encoding_example():
	enc = preprocessing.OneHotEncoder()

# Imputation of missing values.
def imputation_example():
	raise NotImplementedError

# Generating polynomial features.
def polynomial_feature_example():
	raise NotImplementedError

# Custom transformation.
def custom_transformation_example():
	raise NotImplementedError

def main():
	standardization_example()
	normalization_example()
	whitening_example()

	binarization_example()
	encoding_example()

	imputation_example()

	#polynomial_feature_example()
	#custom_transformation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
