#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.scipy.org/doc/scipy-0.18.1/reference/misc.html

import numpy as np
import scipy.misc

def simple_example():
	img_filename = 'D:/dataset/pattern_recognition/street1.png'
	#img_filename = 'D:/dataset/pattern_recognition/street2.jpg'

	img_arr = scipy.misc.imread(img_filename, mode='RGB')
	#img_arr = np.tile(np.arange(255), (255, 1))

	img_arr_rotated = scipy.misc.imrotate(img_arr, angle=45, interp='bilinear')
	img_arr_filtered = scipy.misc.imfilter(img_arr, ftype='edge_enhance_more')

	scipy.misc.imshow(img_arr_rotated)

	scipy.misc.imsave('tmp.jpg', img_arr)

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
