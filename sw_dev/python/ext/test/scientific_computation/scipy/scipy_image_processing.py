#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/scipy-0.18.1/reference/misc.html

import numpy as np
import scipy.ndimage, scipy.misc

def morphological_operation_test():
	img_filename = 'D:/dataset/phenotyping/RDA/all_plants_foreground/adaptor1_side_120_foreground.png'

	img = scipy.misc.imread(img_filename, mode='L')

	img_eroded = scipy.ndimage.grey_erosion(img, size=(3, 3))

	#footprint = scipy.ndimage.generate_binary_structure(2, 2)
	#img_eroded = scipy.ndimage.grey_erosion(img, size=(3, 3), footprint=footprint)

	#scipy.misc.imshow(img_eroded)
	scipy.misc.imsave('tmp.jpg', img_eroded)

def main():
	morphological_operation_test()

	# Hough transform.
	# REF [function] >> estimate_page_orientation_based_on_fft() in ${DataAnalysis_HOME}/app/document/estimate_document_orientation.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
