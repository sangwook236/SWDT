#!/usr/bin/env python

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

def main():
	basic_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
