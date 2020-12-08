#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/NVlabs/ocrodeg

import sys
sys.path.append('../../../src')

import os, math, random
import numpy as np
import scipy.ndimage as ndi
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
import pylab
import cv2

# REF [function] >> noise_distort1d() in ${ocrodeg_HOME}/ocrodeg/degrade.py.
def my_noise_distort1d(shape, in_vec, sigma=100.0, magnitude=100.0):
	h, w = shape[:2]
	#noise = ndi.gaussian_filter(pylab.randn(w), sigma)
	noise = ndi.gaussian_filter(in_vec, sigma)
	noise *= magnitude / np.amax(np.abs(noise))
	dys = np.array([noise]*h)
	deltas = np.array([dys, np.zeros((h, w))])
	return deltas

def simple_example():
	#matplotlib.rc('image', cmap='gray', interpolation='bicubic')

	if False:
		image_filepath = './poem_01.jpg'  # A color page.
		patch_hmin, patch_hmax, patch_wmin, patch_wmax = 160, 460, 40, 370
		is_grayscale, is_normalized = False, False
	elif False:
		image_filepath = './W1P0_title.png'  # A grayscale text line.
		patch_hmin, patch_hmax, patch_wmin, patch_wmax = 0, 89, 0, 1323
		is_grayscale, is_normalized = True, False
	else:
		image_filepath = './W1P0.png'  # A grayscale page. np.float32. [0, 1].
		patch_hmin, patch_hmax, patch_wmin, patch_wmax = 1900, 2156, 1000, 1256
		is_grayscale, is_normalized = True, True

	if is_grayscale:
		import ocrodeg  # For grayscale images.
		cmap = 'gray'
	else:
		import my_ocrodeg.degrade as ocrodeg  # For color images.
		cmap = None

	image = plt.imread(image_filepath)
	#image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	#image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if image is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	if not is_normalized: image = image / 255
	print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(image.shape, image.dtype, np.min(image), np.max(image)))

	patch = image[patch_hmin:patch_hmax, patch_wmin:patch_wmax]

	#--------------------
	# Page rotation.

	if True:
		plt.figure('Page rotation')
		for i, angle in enumerate([0, 90, 180, 270]):
			rotated = ndi.rotate(image, angle, reshape=True, mode='nearest')
			#print('Rotated image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(rotated.shape, rotated.dtype, np.min(rotated), np.max(rotated)))

			# TODO [check] >> Which one is better?
			rotated = np.clip(rotated, 0, 1)
			#minv, maxv = np.min(rotated), np.max(rotated)
			#rotated = (rotated - minv) / (maxv - minv)

			plt.subplot(2, 2, i + 1)
			plt.axis('off')
			plt.imshow(rotated, cmap=cmap)

	#--------------------
	# Random geometric transformation.

	if True:
		plt.figure('Geometric transformation - Random transform')
		for i in range(4):
			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.transform_image(image, **ocrodeg.random_transform()), cmap=cmap)

		plt.figure('Geometric transformation - Rotation')
		for i, angle in enumerate([-4, -2, 0, 2]):
			#angle = random.uniform(0, 2 * math.pi)
			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.transform_image(image, angle=angle * math.pi / 180), cmap=cmap)

		plt.figure('Geometric transformation - Anistropic scaling')  # ?
		for i, aniso in enumerate([0.5, 1.0, 1.5, 2.0]):
			#aniso = 10**random.uniform(-0.1, 0.1)
			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.transform_image(image, aniso=aniso), cmap=cmap)

		plt.figure('Geometric transformation - Scaling')
		for i, scale in enumerate([0.5, 0.9, 1.0, 2.0]):
			#scale = 10**random.uniform(-0.1, 0.1)
			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.transform_image(image, scale=scale), cmap=cmap)

		plt.figure('Geometric transformation - Translation')
		for i in range(4):
			translation = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.transform_image(image, translation=translation), cmap=cmap)

	#--------------------
	# Random distortion.

	if True:
		plt.figure('Random distortion')
		for i, sigma in enumerate([1.0, 2.0, 5.0, 20.0]):
			noise = ocrodeg.bounded_gaussian_noise(image.shape[:2], sigma, maxdelta=5.0)
			distorted = ocrodeg.distort_with_noise(image, noise)

			plt.subplot(1, 4, i + 1)
			h, w = image.shape[:2]
			plt.axis('off')
			plt.imshow(distorted[h//2-200:h//2+200, w//3-200:w//3+200], cmap=cmap)

	#--------------------
	# Ruled surface distortion.

	if True:
		plt.figure('Ruled surface distortion')
		for i, mag in enumerate([5.0, 20.0, 100.0, 200.0]):
			noise = ocrodeg.noise_distort1d(image.shape[:2], magnitude=mag)
			distorted = ocrodeg.distort_with_noise(image, noise)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(distorted[:1500], cmap=cmap)

		in_vec = pylab.randn(image.shape[1])
		plt.figure('Ruled surface distortion - No random input')
		#for i, mag in enumerate([5.0, 200.0, 200.0, 200.0]):
		for i, mag in enumerate([50.0, 100.0, 150.0, 200.0]):
			#noise = ocrodeg.noise_distort1d(image.shape[:2], magnitude=mag)
			noise = my_noise_distort1d(image.shape, in_vec, magnitude=mag)  # My modification.
			distorted = ocrodeg.distort_with_noise(image, noise)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(distorted[:1500], cmap=cmap)

	#--------------------
	# Blur, thresholding, noise.

	if True:
		plt.figure('Blur (ndi)')
		for i, sigma in enumerate([0, 1, 2, 4]):
			blurred = ndi.gaussian_filter(patch, sigma)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(blurred, cmap=cmap)

		plt.figure('Thresholding')
		for i, sigma in enumerate([0, 1, 2, 4]):
			gray = skimage.color.rgb2gray(patch)
			blurred = ndi.gaussian_filter(gray, sigma)
			thresholded = 1.0 * (blurred > 0.5)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(thresholded, cmap='gray')

		plt.figure('Blur - sigma')
		for i, sigma in enumerate([0.0, 1.0, 2.0, 4.0]):
			gray = skimage.color.rgb2gray(patch)
			blurred = ocrodeg.binary_blur(gray, sigma, noise=0.0)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(blurred, cmap='gray')

		plt.figure('Blur - noise')
		for i, noise in enumerate([0.0, 0.1, 0.2, 0.3]):
			gray = skimage.color.rgb2gray(patch)
			blurred = ocrodeg.binary_blur(gray, sigma=2.0, noise=noise)

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(blurred, cmap='gray')

	#--------------------
	# Random blob.

	if True:
		plt.figure('Random blob')
		for i, sz in enumerate([2, 5, 10, 20]):
			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(ocrodeg.random_blobs(patch.shape[:2], blobdensity=3e-4, size=sz, roughness=2.0), cmap='gray')

		blotched = ocrodeg.random_blotches(patch, fgblobs=3e-4, bgblobs=1e-4, fgscale=10, bgscale=10)
		#blotched = np.minimum(np.maximum(patch, ocrodeg.random_blobs(patch.shape, 30, 10)), 1 - ocrodeg.random_blobs(patch.shape, 15, 8))

		plt.figure('Random blotch')
		plt.subplot(121); plt.axis('off'); plt.imshow(patch, cmap=cmap)
		plt.subplot(122); plt.axis('off'); plt.imshow(blotched, cmap=cmap)

	#--------------------
	# Multiscale noise.

	if True:
		plt.figure('Multiscale noise')
		for i in range(4):
			noisy = ocrodeg.make_multiscale_noise_uniform((512, 512), srange=(1.0, 100.0), nscales=4, limits=(0.0, 1.0))

			plt.subplot(1, 4, i + 1)
			plt.axis('off')
			plt.imshow(noisy, vmin=0, vmax=1, cmap='gray')

		# Foreground/background selection.
		plt.figure('Multiscale noise - Background')
		plt.subplot(121); plt.axis('off'); plt.imshow(patch, cmap=cmap)
		plt.subplot(122); plt.axis('off'); plt.imshow(ocrodeg.printlike_multiscale(patch, blur=1.0, blotches=5e-05), cmap=cmap)

	#--------------------
	# Fibrous noise.

	if True:
		plt.figure('Fibrous noise')
		plt.axis('off')
		plt.imshow(ocrodeg.make_fibrous_image((256, 256), nfibers=700, l=300, a=0.01, stepsize=0.5, limits=(0.1, 1.0), blur=1.0), cmap='gray')

		# Foreground/background selection.
		plt.figure('Fibrous noise - Background')
		plt.subplot(121); plt.axis('off'); plt.imshow(patch, cmap=cmap)
		plt.subplot(122); plt.axis('off'); plt.imshow(ocrodeg.printlike_fibrous(patch, blur=1.0, blotches=5e-05), cmap=cmap)

	plt.show()

def paired_image_mask_example():
	import ocrodeg  # For grayscale images.
	import my_ocrodeg.degrade as ocrodeg_rgb  # For color images.

	image_height, image_width = 500, 500
	image = np.zeros((image_height, image_width, 3), dtype=np.float32)
	mask = np.zeros((image_height, image_width), dtype=np.float32)

	cv2.rectangle(image, (100, 100), (200, 200), (0, 0, 1), cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(image, (300, 100), (400, 200), (0, 1, 0), cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(image, (100, 300), (200, 400), (1, 0, 0), cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(image, (300, 300), (400, 400), (1, 0, 1), cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(image, (200, 200), (300, 300), (1, 1, 1), cv2.FILLED, cv2.LINE_8)

	cv2.rectangle(mask, (100, 100), (200, 200), 1, cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(mask, (300, 100), (400, 200), 1, cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(mask, (100, 300), (200, 400), 1, cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(mask, (300, 300), (400, 400), 1, cv2.FILLED, cv2.LINE_8)
	cv2.rectangle(mask, (200, 200), (300, 300), 1, cv2.FILLED, cv2.LINE_8)

	print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(image.shape, image.dtype, np.min(image), np.max(image)))
	print('Mask: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mask.shape, mask.dtype, np.min(mask), np.max(mask)))

	cv2.imshow('Image', image)
	cv2.imshow('Mask', mask)

	if False:
		# Page rotation.
		angle = random.uniform(-180, 180)

		image = ndi.rotate(image, angle, reshape=True, mode='nearest')
		# TODO [check] >> Which one is better?
		image = np.clip(image, 0, 1)
		#minv, maxv = np.min(image), np.max(image)
		#image = (image - minv) / (maxv - minv)

		mask = ndi.rotate(mask, angle, reshape=True, mode='nearest')
		# TODO [check] >> Which one is better?
		mask = np.clip(mask, 0, 1)
		#minv, maxv = np.min(mask), np.max(mask)
		#mask = (mask - minv) / (maxv - minv)
	elif True:
		# Random geometric transformation.
		transform_kwargs = ocrodeg.random_transform()
		image = ocrodeg_rgb.transform_image(image, **transform_kwargs)
		mask = ocrodeg.transform_image(mask, **transform_kwargs)
	elif False:
		# Random distortion.
		sigma = random.uniform(1, 20)
		noise = ocrodeg.bounded_gaussian_noise(image.shape[:2], sigma, maxdelta=5.0)  # Shape = (2, image height, image width).
		image = ocrodeg_rgb.distort_with_noise(image, noise.copy())
		mask = ocrodeg.distort_with_noise(mask, noise.copy())
	elif False:
		# Ruled surface distortion.
		mag = random.uniform(5, 200)
		#noise = ocrodeg.noise_distort1d(image.shape[:2], magnitude=mag)
		in_vec = pylab.randn(image.shape[1])
		noise = my_noise_distort1d(image.shape, in_vec, magnitude=mag)  # My modification. Shape = (2, image height, image width).
		image = ocrodeg_rgb.distort_with_noise(image, noise.copy())
		mask = ocrodeg.distort_with_noise(mask, noise.copy())
	elif False:
		# Blur, thresholding, noise.
		sigma = random.uniform(0.5, 1)
		image = ndi.gaussian_filter(image, sigma)
		#mask = ndi.gaussian_filter(mask, sigma)

		"""
		sigma = random.uniform(0, 4)
		image = skimage.color.rgb2gray(image)
		image = ndi.gaussian_filter(image, sigma)
		image = 1.0 * (image > 0.5)
		##mask = skimage.color.rgb2gray(mask)
		#mask = ndi.gaussian_filter(mask, sigma)
		#mask = 1.0 * (mask > 0.5)
		"""

		"""
		sigma = random.uniform(0, 4)
		image = skimage.color.rgb2gray(image)
		image = ocrodeg.binary_blur(image, sigma, noise=0.0)
		##mask = skimage.color.rgb2gray(mask)
		#mask = ocrodeg.binary_blur(mask, sigma, noise=0.0)
		"""

		"""
		noise = random.uniform(0, 0.3)
		image = skimage.color.rgb2gray(image)
		image = ocrodeg.binary_blur(image, sigma=2.0, noise=noise)
		##mask = skimage.color.rgb2gray(mask)
		#mask = ocrodeg.binary_blur(mask, sigma=2.0, noise=noise)
		"""
	elif False:
		# Random blob.
		image = ocrodeg_rgb.random_blotches(image, fgblobs=3e-4, bgblobs=1e-4, fgscale=10, bgscale=10)
		#image = np.minimum(np.maximum(image, ocrodeg.random_blobs(image.shape[:2], 30, 10)), 1 - ocrodeg.random_blobs(image.shape[:2], 15, 8))
		#mask = ocrodeg.random_blotches(mask, fgblobs=3e-4, bgblobs=1e-4, fgscale=10, bgscale=10)
		#mask = np.minimum(np.maximum(mask, ocrodeg.random_blobs(mask.shape, 30, 10)), 1 - ocrodeg.random_blobs(mask.shape, 15, 8))
	elif False:
		# Multiscale noise.
		image = ocrodeg_rgb.printlike_multiscale(image, blur=1.0, blotches=5e-05)
		#mask = ocrodeg.printlike_multiscale(mask, blur=1.0, blotches=5e-05)
	elif False:
		# Fibrous noise.
		image = ocrodeg_rgb.printlike_fibrous(image, blur=1.0, blotches=5e-05)
		#mask = ocrodeg.printlike_fibrous(mask, blur=1.0, blotches=5e-05)

	cv2.imshow('Distorted Image', image)
	cv2.imshow('Distorted Mask', mask)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	#simple_example()
	paired_image_mask_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
