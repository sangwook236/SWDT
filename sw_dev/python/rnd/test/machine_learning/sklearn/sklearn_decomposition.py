#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import sklearn
import sklearn.decomposition

def pca_test():
	X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	pca = sklearn.decomposition.PCA(n_components=2)
	pca.fit(X)

	print(pca.explained_variance_ratio_)
	print(pca.singular_values_)

	pca = sklearn.decomposition.PCA(n_components=2, svd_solver='full')
	pca.fit(X)

	print(pca.explained_variance_ratio_)
	print(pca.singular_values_)

	pca = sklearn.decomposition.PCA(n_components=1, svd_solver='arpack')
	pca.fit(X)

	print(pca.explained_variance_ratio_)
	print(pca.singular_values_)

def main():
	pca_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
