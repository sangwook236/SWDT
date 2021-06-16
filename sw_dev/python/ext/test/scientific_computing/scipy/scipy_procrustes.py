#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html

import numpy as np
import scipy.spatial

# REF [paper] >> "Manifold Alignment using Procrustes Analysis", ICML 2008.
#	Data correspondence problem has to be solved. (?)
def procrustes(X, Y):
	"""
	X: M x 2 matrix of reference points.
	Y: N x 2 matrix of points to be aligned.
	"""

	X_mean = np.mean(X)
	#X -= X_mean
	X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0)) * 2 - 1
	#Y -= np.mean(Y)
	Y = (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0)) * 2 - 1

	U, S, Vh = np.linalg.svd(Y.T @ X)
	Q = U @ Vh
	k = np.sum(S) / np.trace(Y.T @ Y)

	Y_aligned = k * Y @ Q
	return X, Y_aligned
	#return Y_aligned + X_mean
	#return Y_aligned + Y_mean

def main():
	a = np.array([[1, 3], [1, 2], [1, 1], [2, 1]], 'd')
	b = np.array([[4, -2], [4, -4], [4, -6], [2, -6]], 'd')

	mtx1, mtx2, disparity = scipy.spatial.procrustes(a, b)
	print('a =\n', mtx1)
	print('b =\n', mtx2)
	print('disparity =', disparity)

	a_transformed, b_aligned = procrustes(a, a)
	print('a =\n', a_transformed)
	print('b =\n', b_aligned)

	#--------------------
	X = np.loadtxt('./fish_target.txt')
	Y = np.loadtxt('./fish_source.txt')

	mtx1, mtx2, disparity = scipy.spatial.procrustes(X, Y)
	print('X =\n', mtx1)
	print('Y =\n', mtx2)
	print('disparity =', disparity)

	X_transformed, Y_aligned = procrustes(X, Y)
	print('X =\n', X_transformed)
	print('Y =\n', Y_aligned)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
