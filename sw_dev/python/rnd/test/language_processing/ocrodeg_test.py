#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/NVlabs/ocrodeg

import sys
sys.path.append('../../../src')

import os, math, random
import numpy as np
import scipy.ndimage as ndi
import ocrodeg
import cv2
import pylab
import matplotlib.pyplot as plt

# REF [function] >> noise_distort1d() in ${ocrodeg_HOME}/ocrodeg/degrade.py.
def my_noise_distort1d(shape, in_vec, sigma=100.0, magnitude=100.0):
	h, w = shape
	#noise = ndi.gaussian_filter(pylab.randn(w), sigma)
	noise = ndi.gaussian_filter(in_vec, sigma)
	noise *= magnitude / np.amax(abs(noise))
	dys = np.array([noise]*h)
	deltas = np.array([dys, np.zeros((h, w))])
	return deltas

def simple_example():
	image_filepath = './W1P0.png'
	image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	if image is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return

	#--------------------
	# Page rotation.

	plt.figure('Page rotation')
	for i, angle in enumerate([0, 90, 180, 270]):
		plt.subplot(2, 2, i + 1)
		plt.imshow(ndi.rotate(image, angle), cmap='gray')

	#--------------------
	# Random geometric transformation.

	plt.figure('Geometric transformation - Random transform')
	for i in range(4):
		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.transform_image(image, **ocrodeg.random_transform()), cmap='gray')

	plt.figure('Geometric transformation - Rotation')
	for i, angle in enumerate([-4, -2, 0, 2]):
		#angle = random.uniform(0, 2 * math.pi)
		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.transform_image(image, angle=angle * math.pi / 180), cmap='gray')

	plt.figure('Geometric transformation - Anistropic scaling')  # ?
	for i, aniso in enumerate([0.5, 1.0, 1.5, 2.0]):
		#aniso = 10**random.uniform(-0.1, 0.1)
		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.transform_image(image, aniso=aniso), cmap='gray')

	plt.figure('Geometric transformation - Scaling')
	for i, scale in enumerate([0.5, 0.9, 1.0, 2.0]):
		#scale = 10**random.uniform(-0.1, 0.1)
		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.transform_image(image, scale=scale), cmap='gray')

	plt.figure('Geometric transformation - Translation')
	for i in range(4):
		translation = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.transform_image(image, translation=translation), cmap='gray')

	#--------------------
	# Random distortion.

	plt.figure('Random distortion')
	for i, sigma in enumerate([1.0, 2.0, 5.0, 20.0]):
		noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, maxdelta=5.0)
		distorted = ocrodeg.distort_with_noise(image, noise)

		plt.subplot(2, 2, i + 1)
		h, w = image.shape
		plt.imshow(distorted[h//2-200:h//2+200, w//3-200:w//3+200], cmap='gray')

	#--------------------
	# Ruled surface distortion.

	plt.figure('Ruled surface distortion')
	for i, mag in enumerate([5.0, 20.0, 100.0, 200.0]):
		noise = ocrodeg.noise_distort1d(image.shape, magnitude=mag)
		distorted = ocrodeg.distort_with_noise(image, noise)

		plt.subplot(2, 2, i + 1)
		plt.imshow(distorted[:1500], cmap='gray')

	in_vec = pylab.randn(image.shape[1])
	plt.figure('Ruled surface distortion - No random noise')
	for i, mag in enumerate([5.0, 200.0, 200.0, 200.0]):
		#noise = ocrodeg.noise_distort1d(image.shape, magnitude=mag)
		noise = my_noise_distort1d(image.shape, in_vec, magnitude=mag)  # My modification.
		distorted = ocrodeg.distort_with_noise(image, noise)

		plt.subplot(2, 2, i + 1)
		plt.imshow(distorted[:1500], cmap='gray')

	#--------------------
	# Blur, thresholding, noise.

	patch = image[1900:2156, 1000:1256] / 255

	plt.figure('Blur (ndi)')
	for i, sigma in enumerate([0, 1, 2, 4]):
		blurred = ndi.gaussian_filter(patch, sigma)

		plt.subplot(2, 2, i + 1)
		plt.imshow(blurred, cmap='gray')

	plt.figure('Thresholding')
	for i, sigma in enumerate([0, 1, 2, 4]):
		blurred = ndi.gaussian_filter(patch, sigma)
		thresholded = 1.0 * (blurred > 0.5)

		plt.subplot(2, 2, i + 1)
		plt.imshow(thresholded, cmap='gray')

	plt.figure('Blur - sigma')
	for i, sigma in enumerate([0.0, 1.0, 2.0, 4.0]):
		blurred = ocrodeg.binary_blur(patch, sigma, noise=0.0)

		plt.subplot(2, 2, i + 1)
		plt.imshow(blurred, cmap='gray')

	plt.figure('Blur - noise')
	for i, noise in enumerate([0.0, 0.1, 0.2, 0.3]):
		blurred = ocrodeg.binary_blur(patch, sigma=2.0, noise=noise)

		plt.subplot(2, 2, i + 1)
		plt.imshow(blurred, cmap='gray')

	#--------------------
	# Random blob.

	plt.figure('Random blob')
	for i, sz in enumerate([2, 5, 10, 20]):
		plt.subplot(2, 2, i + 1)
		plt.imshow(ocrodeg.random_blobs(patch.shape, blobdensity=3e-4, size=sz, roughness=2.0), cmap='gray')

	blotched = ocrodeg.random_blotches(patch, fgblobs=3e-4, bgblobs=1e-4, fgscale=10, bgscale=10)
	#blotched = np.minimum(np.maximum(patch, ocrodeg.random_blobs(patch.shape, 30, 10)), 1 - ocrodeg.random_blobs(patch.shape, 15, 8))

	plt.figure('Random blotch')
	plt.subplot(121); plt.imshow(patch, cmap='gray')
	plt.subplot(122); plt.imshow(blotched, cmap='gray')

	#--------------------
	# Multiscale noise.

	plt.figure('Multiscale noise')
	for i in range(4):
		noisy = ocrodeg.make_multiscale_noise_uniform((512, 512), srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0))

		plt.subplot(2, 2, i + 1)
		plt.imshow(noisy, vmin=0, vmax=1, cmap='gray')

	# Foreground/background selection.
	plt.figure('Multiscale noise - Background')
	plt.subplot(121); plt.imshow(patch, cmap='gray')
	plt.subplot(122); plt.imshow(ocrodeg.printlike_multiscale(patch, blur=1.0, blotches=5e-05), cmap='gray')

	#--------------------
	# Fibrous noise.

	plt.figure('Fibrous noise')
	plt.imshow(ocrodeg.make_fibrous_image((256, 256), nfibers=700, l=300, a=0.01, stepsize=0.5, limits=(0.1, 1.0), blur=1.0), cmap='gray')

	# Foreground/background selection.
	plt.figure('Fibrous noise - Background')
	plt.subplot(121); plt.imshow(patch, cmap='gray')
	plt.subplot(122); plt.imshow(ocrodeg.printlike_fibrous(patch, blur=1.0, blotches=5e-05), cmap='gray')

	plt.show()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
