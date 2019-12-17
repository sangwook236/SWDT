#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import skimage
import skimage.filters, skimage.morphology
import matplotlib
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
def try_all_threshold_example():
	img = skimage.data.page()

	# Specify a radius for local thresholding algorithms.
	# If it is not specified, only global algorithms are called.
	fig, ax = skimage.filters.try_all_threshold(img, figsize=(10, 8), verbose=False)
	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_thresholding.html
def threshold_example():
	image = skimage.data.camera()

	#thresh = skimage.filters.threshold_isodata(image)
	#thresh = skimage.filters.threshold_li(image)
	#thresh = skimage.filters.threshold_mean(image)
	#thresh = skimage.filters.threshold_minimum(image)
	thresh = skimage.filters.threshold_otsu(image)
	#thresh = skimage.filters.threshold_triangle(image)
	#thresh = skimage.filters.threshold_yen(image)
	binary = image > thresh

	fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
	ax = axes.ravel()
	ax[0] = plt.subplot(1, 3, 1)
	ax[1] = plt.subplot(1, 3, 2)
	ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

	ax[0].imshow(image, cmap=plt.cm.gray)
	ax[0].set_title('Original')
	ax[0].axis('off')

	ax[1].hist(image.ravel(), bins=256)
	ax[1].set_title('Histogram')
	ax[1].axvline(thresh, color='r')

	ax[2].imshow(binary, cmap=plt.cm.gray)
	ax[2].set_title('Thresholded')
	ax[2].axis('off')

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_multiotsu.html
def multiotsu_example():
	# Setting the font size for all plots.
	matplotlib.rcParams['font.size'] = 9

	# The input image.
	image = skimage.data.camera()

	# Applying multi-Otsu threshold for the default value, generating three classes.
	thresholds = skimage.filters.threshold_multiotsu(image)

	# Using the threshold values, we generate the three regions.
	regions = np.digitize(image, bins=thresholds)
	#regions_colorized = skimage.color.label2rgb(regions)
	#plt.imshow(regions_colorized)

	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

	# Plotting the original image.
	ax[0].imshow(image, cmap='gray')
	ax[0].set_title('Original')
	ax[0].axis('off')

	# Plotting the histogram and the two thresholds obtained from multi-Otsu.
	ax[1].hist(image.ravel(), bins=255)
	ax[1].set_title('Histogram')
	for thresh in thresholds:
		ax[1].axvline(thresh, color='r')

	# Plotting the Multi Otsu result.
	ax[2].imshow(regions, cmap='Accent')
	ax[2].set_title('Multi-Otsu result')
	ax[2].axis('off')

	plt.subplots_adjust()

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/applications/plot_thresholding.html
def local_otsu_threshold_example():
	# Otsu's threshold method can be applied locally.
	# For each pixel, an "optimal" threshold is determined by maximizing the variance between two classes of pixels of the local neighborhood defined by a structuring element.

	img = skimage.util.img_as_ubyte(skimage.data.page())

	radius = 15
	selem = skimage.morphology.disk(radius)

	local_otsu = skimage.filters.rank.otsu(img, selem)
	threshold_global_otsu = skimage.filters.threshold_otsu(img)
	global_otsu = img >= threshold_global_otsu

	fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
	ax = axes.ravel()
	plt.tight_layout()

	fig.colorbar(ax[0].imshow(img, cmap=plt.cm.gray), ax=ax[0], orientation='horizontal')
	ax[0].set_title('Original')
	ax[0].axis('off')

	fig.colorbar(ax[1].imshow(local_otsu, cmap=plt.cm.gray), ax=ax[1], orientation='horizontal')
	ax[1].set_title('Local Otsu (radius=%d)' % radius)
	ax[1].axis('off')

	ax[2].imshow(img >= local_otsu, cmap=plt.cm.gray)
	ax[2].set_title('Original >= Local Otsu' % threshold_global_otsu)
	ax[2].axis('off')

	ax[3].imshow(global_otsu, cmap=plt.cm.gray)
	ax[3].set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
	ax[3].axis('off')

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/applications/plot_thresholding.html
def local_threshold_example():
	image = skimage.data.page()

	global_thresh = skimage.filters.threshold_otsu(image)
	binary_global = image > global_thresh

	block_size = 35
	local_thresh = skimage.filters.threshold_local(image, block_size, offset=10)
	binary_local = image > local_thresh

	fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
	ax = axes.ravel()
	plt.gray()

	ax[0].imshow(image)
	ax[0].set_title('Original')

	ax[1].imshow(binary_global)
	ax[1].set_title('Global thresholding')

	ax[2].imshow(binary_local)
	ax[2].set_title('Local thresholding')

	for a in ax:
		a.axis('off')

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_niblack_sauvola.html
def niblack_and_sauvola_example():
	matplotlib.rcParams['font.size'] = 9

	image = skimage.data.page()

	binary_global = image > skimage.filters.threshold_otsu(image)

	window_size = 25
	thresh_niblack = skimage.filters.threshold_niblack(image, window_size=window_size, k=0.8)
	thresh_sauvola = skimage.filters.threshold_sauvola(image, window_size=window_size)

	binary_niblack = image > thresh_niblack
	binary_sauvola = image > thresh_sauvola

	plt.figure(figsize=(8, 7))
	plt.subplot(2, 2, 1)
	plt.imshow(image, cmap=plt.cm.gray)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(2, 2, 2)
	plt.title('Global Threshold')
	plt.imshow(binary_global, cmap=plt.cm.gray)
	plt.axis('off')

	plt.subplot(2, 2, 3)
	plt.imshow(binary_niblack, cmap=plt.cm.gray)
	plt.title('Niblack Threshold')
	plt.axis('off')

	plt.subplot(2, 2, 4)
	plt.imshow(binary_sauvola, cmap=plt.cm.gray)
	plt.title('Sauvola Threshold')
	plt.axis('off')

	plt.show()

def main():
	#try_all_threshold_example()

	#threshold_example()
	#multiotsu_example()

	# Local/adaptive thresholding.
	local_otsu_threshold_example()
	#local_threshold_example()
	#niblack_and_sauvola_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
