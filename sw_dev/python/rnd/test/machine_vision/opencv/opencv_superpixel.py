#!/usr/bin/env python

import cv2 as cv

def simple_example():
	#image_filepath = '../../../data/machine_vision/build.png'
	image_filepath = 'D:/work_biz/silicon_minds/DataAnalysis_bitbucket/data/id_images/rrc_00.jpg'

	# Read gray image.
	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	if True:
		# Linear Spectral Clustering (LSC) superpixels algorithm.
		superpixel = cv.ximgproc.createSuperpixelLSC(img, region_size=20, ratio=0.075)

		# Calculate the superpixel segmentation.
		superpixel.iterate(num_iterations=10)

		#superpixel.enforceLabelConnectivity(min_element_size=20)
	elif False:
		# Superpixels Extracted via Energy-Driven Sampling (SEEDS) superpixels algorithm.
		superpixel = cv.ximgproc.createSuperpixelSEEDS(
			img.shape[1], img.shape[0], img.shape[2],
			num_superpixels = 200, num_levels=3, prior=2, histogram_bins=5, double_step=False
		)

		# Calculate the superpixel segmentation.
		superpixel.iterate(img, num_iterations=10)
	elif False:
		# Linear Spectral Clustering (LSC) superpixels algorithm.
		superpixel = cv.ximgproc.createSuperpixelLSC(img, algorithm=cv.ximgproc.SLICO, region_size=20, ruler	=10.0)  # { SLIC, SLICO, MSLIC }.

		# Calculate the superpixel segmentation.
		superpixel.iterate(num_iterations=10)

		#superpixel.enforceLabelConnectivity(min_element_size=20)

	print('#superpixels =', superpixel.getNumberOfSuperpixels())

	superpixel_label = superpixel.getLabels()  # CV_32UC1. [0, getNumberOfSuperpixels()].
	superpixel_contour_mask = superpixel.getLabelContourMask(thick_line=True)  # CV_8UC1.

	img[superpixel_contour_mask > 0] = (0, 0, 255)
	cv.imshow('Superpixel', img)
	cv.waitKey(0)

	cv.destroyAllWindows()

def main():
	#simple_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
