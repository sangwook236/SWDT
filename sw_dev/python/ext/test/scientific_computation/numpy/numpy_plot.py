#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#histograms

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(h, w, maxit = 20):
	"""Returns an image of the Mandelbrot fractal of size (h,w)."""
	y, x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j]
	c = x + y * 1j
	z = c
	divtime = maxit + np.zeros(z.shape, dtype = int)

	for i in range(maxit):
		z = z**2 + c
		diverge = z * np.conj(z) > 2**2  # Who is diverging.
		div_now = diverge & (divtime == maxit)  # Who is diverging now.
		divtime[div_now] = i  # Note when.
		z[diverge] = 2  # Avoid diverging too much.

	return divtime

def fractal_example():
	plt.imshow(mandelbrot(400, 400))
	plt.show()

def histogram_example():
	# Build a vector of 10000 normal deviates with variance 0.5^2 and mean 2.
	mu, sigma = 2, 0.5
	v = np.random.normal(mu, sigma, 10000)

	# Plot a normalized histogram with 50 bins.
	plt.hist(v, bins = 50, normed = 1)  # Matplotlib version (plot).
	plt.show()

	# Compute the histogram with numpy and then plot it.
	(n, bins) = np.histogram(v, bins = 50, normed = True)  # NumPy version (no plot).
	plt.plot(.5 * (bins[1:] + bins[:-1]), n)
	plt.show()

def main():
	fractal_example()
	histogram_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
