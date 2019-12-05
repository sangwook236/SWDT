#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import cv2 as cv

def drawAxis(img, p_, q_, colour, scale):
	p = list(p_)
	q = list(q_)

	angle = math.atan2(p[1] - q[1], p[0] - q[0])  # angle in radians.
	hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

	# Here we lengthen the arrow by a factor of scale.
	q[0] = p[0] - scale * hypotenuse * math.cos(angle)
	q[1] = p[1] - scale * hypotenuse * math.sin(angle)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

	# Create the arrow hooks.
	p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
	p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

	p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
	p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
	cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

def getOrientation(pts, img, gray):
	# Construct a buffer used by the PCA analysis.
	if False:
		# Uses points on contour.
		"""
		sz = len(pts)
		data_pts = np.empty((sz, 2), dtype=np.float64)
		for i in range(data_pts.shape[0]):
			data_pts[i,0] = pts[i,0,0]
			data_pts[i,1] = pts[i,0,1]
		"""
		data_pts = np.reshape(pts, [pts.shape[0], -1]).astype(np.float64)
	elif False:
		# Uses all points in contour.
		mask = np.zeros(img.shape[:2], dtype=np.uint8)
		cv.drawContours(mask, [pts], 0, (255, 255, 255), cv.FILLED)
		data_pts = cv.findNonZero(mask)
		data_pts = np.reshape(data_pts, [data_pts.shape[0], -1]).astype(np.float64)
	else:
		# Uses non-zero points in contour.
		mask = np.zeros(img.shape[:2], dtype=np.uint8)
		cv.drawContours(mask, [pts], 0, (255, 255, 255), cv.FILLED)
		mask = np.where(mask > 0, gray, mask)
		data_pts = cv.findNonZero(mask)
		data_pts = np.reshape(data_pts, [data_pts.shape[0], -1]).astype(np.float64)

	# Perform PCA analysis.
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)

	# Store the center of the object.
	cntr = (int(mean[0,0]), int(mean[0,1]))

	# Draw the principal components.
	cv.circle(img, cntr, 3, (255, 0, 255), 2)
	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
	drawAxis(img, cntr, p1, (0, 255, 0), 1)
	drawAxis(img, cntr, p2, (255, 255, 0), 5)

	angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0])  # Orientation in radians.

	return angle

# REF [site] >> https://docs.opencv.org/4.1.0/d1/dee/tutorial_introduction_to_pca.html
def simple_pca_example():
	image_filepath = '../../../data/machine_vision/pca_test1.jpg'

	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		print('Could not open or find the image: ', args.input)
		exit(0)

	cv.imshow('Input', img)

	# Convert image to grayscale.
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# Convert image to binary.
	_, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

	# Find all the contours in the thresholded image.
	_, contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

	for i, c in enumerate(contours):
		# Calculate the area of each contour.
		area = cv.contourArea(c);
		# Ignore contours that are too small or too large.
		if area < 1e2 or 1e5 < area:
			continue

		# Draw each contour only for visualisation purposes.
		cv.drawContours(img, contours, i, (0, 0, 255), 2)
		# Find the orientation of each shape.
		getOrientation(c, img, bw)

	cv.imshow('Output', img)
	cv.waitKey(0)

	cv.destroyAllWindows()

def main():
	simple_pca_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
