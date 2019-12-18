#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
from scipy import ndimage as ndi
import skimage, skimage.filters
import matplotlib.pyplot as plt

def compute_feats(image, kernels):
	feats = np.zeros((len(kernels), 2), dtype=np.double)
	for k, kernel in enumerate(kernels):
		filtered = ndi.convolve(image, kernel, mode='wrap')
		feats[k, 0] = filtered.mean()
		feats[k, 1] = filtered.var()
	return feats

def match(feats, ref_feats):
	min_error = np.inf
	min_i = None
	for i in range(ref_feats.shape[0]):
		error = np.sum((feats - ref_feats[i, :])**2)
		if error < min_error:
			min_error = error
			min_i = i
	return min_i

def power(image, kernel):
	# Normalize images for better comparison.
	image = (image - image.mean()) / image.std()
	return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html
def classify_texture_based_on_Gabor_filter_banks():
	# Prepare filter bank kernels.
	kernels = []
	for theta in range(4):
		theta = theta / 4. * np.pi
		for sigma in (1, 3):
			for frequency in (0.05, 0.25):
				kernel = np.real(skimage.filters.gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
				kernels.append(kernel)

	shrink = (slice(0, None, 3), slice(0, None, 3))
	brick = skimage.util.img_as_float(skimage.data.brick())[shrink]
	grass = skimage.util.img_as_float(skimage.data.grass())[shrink]
	gravel = skimage.util.img_as_float(skimage.data.gravel())[shrink]
	image_names = ('brick', 'grass', 'gravel')
	images = (brick, grass, gravel)

	# Prepare reference features.
	ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
	ref_feats[0, :, :] = compute_feats(brick, kernels)
	ref_feats[1, :, :] = compute_feats(grass, kernels)
	ref_feats[2, :, :] = compute_feats(gravel, kernels)

	print('Rotated images matched against references using Gabor filter banks:')

	print('original: brick, rotated: 30deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(brick, angle=190, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])

	print('original: brick, rotated: 70deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(brick, angle=70, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])

	print('original: grass, rotated: 145deg, match result: ', end='')
	feats = compute_feats(ndi.rotate(grass, angle=145, reshape=False), kernels)
	print(image_names[match(feats, ref_feats)])

	# Plot a selection of the filter bank kernels and their responses.
	results = []
	kernel_params = []
	for theta in (0, 1):
		theta = theta / 4. * np.pi
		for frequency in (0.1, 0.4):
			kernel = skimage.filters.gabor_kernel(frequency, theta=theta)
			params = 'theta=%d,\nfrequency=%.2f' % (theta * 180 / np.pi, frequency)
			kernel_params.append(params)
			# Save kernel and the power image for each image.
			results.append((kernel, [power(img, kernel) for img in images]))

	fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(5, 6))
	plt.gray()

	fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

	axes[0][0].axis('off')

	# Plot original images.
	for label, img, ax in zip(image_names, images, axes[0][1:]):
		ax.imshow(img)
		ax.set_title(label, fontsize=9)
		ax.axis('off')

	for label, (kernel, powers), ax_row in zip(kernel_params, results, axes[1:]):
		# Plot Gabor kernel.
		ax = ax_row[0]
		ax.imshow(np.real(kernel))
		ax.set_ylabel(label, fontsize=7)
		ax.set_xticks([])
		ax.set_yticks([])

		# Plot Gabor responses with the contrast normalized for each filter.
		vmin = np.min(powers)
		vmax = np.max(powers)
		for patch, ax in zip(powers, ax_row[1:]):
			ax.imshow(patch, vmin=vmin, vmax=vmax)
			ax.axis('off')

	plt.show()

def main():
	classify_texture_based_on_Gabor_filter_banks()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
