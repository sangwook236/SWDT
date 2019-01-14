#!/usr/bin/env python

import math
import cv2

def line_detection():
	image_filepath = '../../../data/machine_vision/B004_1.jpg'
	#image_filepath = '../../../data/machine_vision/B008_1.jpg'

	# Read gray image.
	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 50, 150, apertureSize=3)

	minLineLength = 500
	maxLineGap = 5
	lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 10, minLineLength, maxLineGap)  # Shape = (lines, 1, 4).

	print('#detected lines =', lines.shape[0])
	for line in lines:
		cv2.line(img, (line[0,0], line[0,1]), (line[0,2], line[0,3]), (0, 255, 0), 2)

	# Show image.
	cv2.imshow('Probabilistic Hough transform', img)
	cv2.waitKey(0)

def main():
	line_detection()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
