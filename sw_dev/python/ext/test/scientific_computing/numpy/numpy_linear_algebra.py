#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#linear-algebra

import numpy as np

def basic_example():
	a = np.array([[1.0, 2.0], [3.0, 4.0]])
	print(a)

	a.transpose()
	np.linalg.inv(a)
	u = np.eye(2)
	j = np.array([[0.0, -1.0], [1.0, 0.0]])

	# Matrix product.
	np.dot(j, j)

	np.trace(u)

	#--------------------
	y = np.array([[5.], [7.]])
	np.linalg.solve(a, y)

	#--------------------
	np.linalg.eig(j)

# REF [site] >> https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
def svd_example():
	a = np.random.randn(9, 6) + 1j * np.random.randn(9, 6)
	b = np.random.randn(2, 7, 8, 3) + 1j * np.random.randn(2, 7, 8, 3)

	u, s, vh = np.linalg.svd(a, full_matrices=True)
	print('u.shape = {}, s.shape = {}, vh.shape = {}.'.format(u.shape, s.shape, vh.shape))
	print('Close? =', np.allclose(a, np.dot(u[:, :6] * s, vh)))
	smat = np.zeros((9, 6), dtype=complex)
	smat[:6, :6] = np.diag(s)
	print('Close? =', np.allclose(a, np.dot(u, np.dot(smat, vh))))

	u, s, vh = np.linalg.svd(a, full_matrices=False)
	print('u.shape = {}, s.shape = {}, vh.shape = {}.'.format(u.shape, s.shape, vh.shape))
	print('Close? =', np.allclose(a, np.dot(u * s, vh)))
	smat = np.diag(s)
	print('Close? =', np.allclose(a, np.dot(u, np.dot(smat, vh))))

	u, s, vh = np.linalg.svd(b, full_matrices=True)
	print('u.shape = {}, s.shape = {}, vh.shape = {}.'.format(u.shape, s.shape, vh.shape))
	print('Close? =', np.allclose(b, np.matmul(u[..., :3] * s[..., None, :], vh)))
	print('Close? =', np.allclose(b, np.matmul(u[..., :3], s[..., None] * vh)))

	u, s, vh = np.linalg.svd(b, full_matrices=False)
	print('u.shape = {}, s.shape = {}, vh.shape = {}.'.format(u.shape, s.shape, vh.shape))
	print('Close? =', np.allclose(b, np.matmul(u * s[..., None, :], vh)))
	print('Close? =', np.allclose(b, np.matmul(u, s[..., None] * vh)))

def main():
	#basic_example()

	svd_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
