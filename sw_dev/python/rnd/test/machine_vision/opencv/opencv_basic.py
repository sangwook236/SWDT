#!/usr/bin/env python

import numpy as np
import cv2

def image_reading_and_writing():
	image_filepath = '../../../data/machine_vision/B004_1.jpg'

	# NOTE [info] >> OpenCV has some difficulty in reading an image file with Hangul path.
	#img = cv2.imread(image_filepath, cv2.IMREAD_UNCHANGED)
	with open(image_filepath, 'rb') as fd:
		bytes = bytearray(fd.read())
		img = cv2.imdecode(np.asarray(bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
	if img is None:
		print('File not found:', img_filepath)
		return

	#cv2.imshow('./saved.png', img)

	# Show image.
	cv2.imshow('Image', img)
	cv2.waitKey(0)

def main():
	image_reading_and_writing()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
