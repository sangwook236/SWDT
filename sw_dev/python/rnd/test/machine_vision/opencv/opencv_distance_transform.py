#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2 as cv

# REF [site] >> https://docs.opencv.org/4.1.0/d2/dbd/tutorial_distance_transform.html
def simple_example():
	#image_filepath = '../../../data/machine_vision/build.png'
	image_filepath = 'D:/work_biz/silicon_minds/DataAnalysis_bitbucket/data/id_images/rrc_00.jpg'

	# Read gray image.
	src = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if src is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	#--------------------
	src[np.all(src == 255, axis=2)] = 0

	# Show output image.
	cv.imshow('Black Background Image', src)

	#--------------------
	kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

	# We need to convert everything in something more deeper then CV_8U because the kernel has some negative values.
	imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
	sharp = np.float32(src)
	imgResult = sharp - imgLaplacian

	# Convert back to 8bits gray scale.
	imgResult = np.clip(imgResult, 0, 255)
	imgResult = imgResult.astype('uint8')
	imgLaplacian = np.clip(imgLaplacian, 0, 255)
	imgLaplacian = np.uint8(imgLaplacian)

	cv.imshow('Laplace Filtered Image', imgLaplacian)
	cv.imshow('New Sharped Image', imgResult)

	#--------------------
	bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
	_, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

	dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

	cv.imshow('Binary Image', bw)
	# Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it.
	cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
	cv.imshow('Distance Transform Image', dist)

	#--------------------
	_, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
	# Dilate a bit the dist image.
	kernel1 = np.ones((3, 3), dtype=np.uint8)
	dist = cv.dilate(dist, kernel1)

	cv.imshow('Peaks', dist)

	dist_8u = dist.astype('uint8')

	# Show image.
	cv.waitKey(0)

	cv.destroyAllWindows()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
