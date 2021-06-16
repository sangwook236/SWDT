#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def draw_multivariate_normal_distribution_test():
	center = [0.5, 0]
	sigma = [1, 2]
	theta = math.radians(60)

	cos_theta, sin_theta = math.cos(theta), math.sin(theta)
	R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
	cov = np.diag(sigma)
	cov = np.matmul(R, np.matmul(cov, R.T))

	rv = scipy.stats.multivariate_normal(center, cov)

	#x, y = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
	x, y = np.mgrid[-1:1:0.01, -1:1:0.01]
	pos = np.dstack((x, y))

	fig = plt.figure()
	plt.contourf(x, y, rv.pdf(pos))
	plt.axes().set_aspect('equal')
	plt.show()

def main():
	draw_multivariate_normal_distribution_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
