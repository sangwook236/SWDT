#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/scipy/reference/fftpack.html

import copy
import numpy as np
import scipy
import cv2
import fft_util

def toy_example():
	Fs = 1000
	N = 1500
	time = np.arange(N) / float(Fs)

	#signal = generate_toy_signal_1(time, noise=False, DC=True)
	signal = fft_util.generate_toy_signal_2(time, noise=False, DC=True)

	fft_util.plot_fft(signal, Fs)
	#fft_util.plot_fft(signal - signal.mean(), Fs)  # Removes DC component.

# REF [site] >> https://docs.opencv.org/2.4/doc/tutorials/core/discrete_fourier_transform/discrete_fourier_transform.html
def image_fft_example():
	image_filepath = './image.png'

	# Read a gray image.
	img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	cv2.imshow('Input', img)

	#--------------------
	# FFT.
	if True:
		img_fft = scipy.fftpack.fft2(img)
	else:
		# Expand input image to optimal size.
		# FIXME [error] >> Some artifacts (vertical and horizontal lines) are generated.
		cols, rows = img.shape[:2]
		m, n = cv2.getOptimalDFTSize(rows), cv2.getOptimalDFTSize(cols)
		img_padded = cv2.copyMakeBorder(img, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, 0)

		img_fft = scipy.fftpack.fft2(img_padded)
		img_fft = img_fft[:cols,:rows]
	img_fft_mag = np.log10(np.abs(img_fft))

	# Crop the spectrum, if it has an odd number of rows or columns.
	img_fft_mag = img_fft_mag[:(img_fft_mag.shape[0] & -2),:(img_fft_mag.shape[1] & -2)]

	# Rearrange the quadrants of Fourier image  so that the origin is at the image center.
	cy, cx = img_fft_mag.shape[0] // 2, img_fft_mag.shape[1] // 2
	top_left = copy.deepcopy(img_fft_mag[:cy,:cx])
	top_right = copy.deepcopy(img_fft_mag[:cy,cx:])
	bottom_left = copy.deepcopy(img_fft_mag[cy:,:cx])
	bottom_right = copy.deepcopy(img_fft_mag[cy:,cx:])
	img_fft_mag[:cy,:cx] = bottom_right
	img_fft_mag[:cy,cx:] = bottom_left
	img_fft_mag[cy:,:cx] = top_right
	img_fft_mag[cy:,cx:] = top_left
	del top_left, top_right, bottom_left, bottom_right

	# Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	img_fft_mag_normalized = cv2.normalize(img_fft_mag, None, 0, 1, cv2.NORM_MINMAX)

	cv2.imshow('FFT', img_fft_mag_normalized)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	#toy_example()

	image_fft_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
