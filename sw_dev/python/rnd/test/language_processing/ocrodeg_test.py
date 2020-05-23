#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../../src')

import os
import numpy as np
import ocrodeg
import cv2
import matplotlib.pyplot as plt

def simple_example():
	image_filepath = './W1P0.png'
	image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	if image is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return

	import pylab
	vec = pylab.randn(image.shape[1])
	for i, mag in enumerate([5.0, 200.0, 200.0, 200.0]):
		plt.subplot(2, 2, i+1)
		#noise = ocrodeg.noise_distort1d(image.shape, magnitude=mag)
		noise = ocrodeg.noise_distort1d2(image.shape, vec, magnitude=mag)  # My modification.
		distorted = ocrodeg.distort_with_noise(image, noise)
		#h, w = image.shape
		plt.imshow(distorted[:1500], cmap='gray')
	plt.show()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
