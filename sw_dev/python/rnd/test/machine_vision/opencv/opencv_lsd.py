#!/usr/bin/env python

import cv2

def main():
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

	cv2.destroyAllWindows()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
