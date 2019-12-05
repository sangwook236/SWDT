#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import cv2

def line_detection():
	image_filepath = '../../../data/machine_vision/sudoku.png'

	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Smooth it, otherwise a lot of false circles may be detected.
	#gray = cv2.pyrUp(cv2.pyrDown(gray))
	#gray = cv2.GaussianBlur(gray, ksize=(9, 9), sigmaX=2, sigmaY=2)
	#gray = cv2.GaussianBlur(gray, (5, 5), 0)
	#gray = cv2.medianBlur(gray, ksize=5)

	edges = cv2.Canny(gray, 50, 200, apertureSize=3)
	#_, edges = cv.threshold(edges, 100, 255, cv.THRESH_BINARY)

	#--------------------
	lines = cv2.HoughLines(edges, rho=1, theta=math.pi / 180, threshold=150, srn=0, stn=0, min_theta=0, max_theta=math.pi)

	if lines is not None:
		print('#detected lines (HoughLines) =', len(lines))
		for line in lines:
			rho, theta = line[0,0], line[0,1]
			a, b = math.cos(theta), math.sin(theta)
			x0, y0 = a * rho, b * rho
			pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
			pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
			cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA);

	#--------------------
	#lines = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=10, minLineLength=500, maxLineGap=5)  # The shape of lines = (#lines, 1, 4).
	lines = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)  # The shape of lines = (#lines, 1, 4).

	if lines is not None:
		print('#detected lines (HoughLinesP) =', len(lines))
		for line in lines:
			cv2.line(img, (line[0,0], line[0,1]), (line[0,2], line[0,3]), (0, 255, 0), 3, cv2.LINE_AA)

	cv2.imshow('Hough transform', img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def circle_detection():
	image_filepath = '../../../data/machine_vision/opencv-logo-white.png'

	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Smooth it, otherwise a lot of false circles may be detected.
	#gray = cv2.pyrUp(cv2.pyrDown(gray))
	#gray = cv2.GaussianBlur(gray, ksize=(9, 9), sigmaX=2, sigmaY=2)
	#gray = cv2.GaussianBlur(gray, (5, 5), 0)
	gray = cv2.medianBlur(gray, ksize=5)

	#circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=2, minDist=gray.shape[0] / 4, param1=200, param2=100, minRadius=0, maxRadius=0)
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

	if circles is not None:
		print('#detected circles =', len(circles[0,:]))
		circles = np.uint16(np.around(circles))
		for circle in circles[0,:]:
			center = (round(circle[0]), round(circle[1]))
			radius = round(circle[2])
			# Draw the circle center.
			cv2.circle(img, center, 3, (0, 255, 0), -1, cv2.LINE_AA, 0)
			# Draw the circle outline.
			cv2.circle(img, center, radius, (0, 0, 255), 3, cv2.LINE_AA, 0)

	cv2.imshow('Circles', img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def main():
	line_detection()
	#circle_detection()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
