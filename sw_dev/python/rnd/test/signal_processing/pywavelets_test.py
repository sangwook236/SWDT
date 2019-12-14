#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/
def simple_example():
	# Load image.
	original = pywt.data.camera()

	# Wavelet transform of image.
	coeffs2 = pywt.dwt2(original, 'bior1.3')

	# Plot approximation and details.
	titles = ['Approximation', ' Horizontal detail', 'Vertical detail', 'Diagonal detail']
	LL, (LH, HL, HH) = coeffs2
	fig = plt.figure(figsize=(12, 3))
	for i, a in enumerate([LL, LH, HL, HH]):
		ax = fig.add_subplot(1, 4, i + 1)
		ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
		ax.set_title(titles[i], fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])

	fig.tight_layout()
	plt.show()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
