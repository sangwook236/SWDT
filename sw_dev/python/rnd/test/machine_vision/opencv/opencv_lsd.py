#!/usr/bin/env python

import cv2

def main():
	image_filepath = '../../../data/machine_vision/B004_1.jpg'
	#image_filepath = '../../../data/machine_vision/B008_1.jpg'

	# Read gray image.
	img = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)

	# Create default parametrization LSD.
	lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV, 0.8)

	# Detect lines in the image.
	lines, width, prec, nfa = lsd.detect(img)  # Lines' shape = (lines, 1, 4).
	print('#detected lines =', lines.shape[0])

	# Draw detected lines in the image.
	drawn_img = lsd.drawSegments(img, lines)

	# Show image.
	cv2.imshow('LSD', drawn_img)
	cv2.waitKey(0)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
