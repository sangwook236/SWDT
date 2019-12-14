#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2

def lsd():
	image_filepath = '../../../data/machine_vision/build.png'

	# Read gray image.
	img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	# Create default parametrization LSD.
	lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, 0.8)

	# Detect lines in the image.
	lines, width, prec, nfa = lsd.detect(img)  # The shape of lines = (#lines, 1, 4).
	print('#detected lines =', lines.shape[0])

	# Draw detected lines in the image.
	drawn_img = lsd.drawSegments(img, lines)

	# Show image.
	cv2.imshow('LSD', drawn_img)
	cv2.waitKey(0)

	#cv2.destroyAllWindows()

def fld():
	image_filepath = '../../../data/machine_vision/build.png'

	# Read gray image.
	img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError('Failed to load an image, {}.'.format(image_filepath))

	# Create default parametrization LSD.
	length_threshold = 10
	distance_threshold = 1.41421356
	canny_th1 = 50.0
	canny_th2 = 50.0
	canny_aperture_size = 3
	do_merge = False
	fld = cv2.ximgproc.createFastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2, canny_aperture_size, )

	# Detect lines in the image.
	lines = fld.detect(img)  # The shape of lines = (#lines, 1, 4).
	print('#detected lines =', lines.shape[0])

	# Draw detected lines in the image.
	drawn_img = fld.drawSegments(img, lines)

	# Show image.
	cv2.imshow('FLD', drawn_img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	lsd()
	fld()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
