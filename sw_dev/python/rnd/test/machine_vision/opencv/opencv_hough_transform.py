#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import cv2

def angle_between_hough_lines(line1, line2):  # line = (rho, theta).
	delta = abs(line1[1] - line2[1])
	return min(delta, math.pi - delta)

def draw_hough_line(image, rho, theta, color, thickness=1):
	# Line equation (the Hesse normal form): x * cos(theta) + y * sin(theta) = rho.
	sin_angle, cos_angle = math.sin(theta), math.cos(theta)
	if math.isclose(sin_angle, 0.0):
		x0 = x1 = round(rho / cos_angle)
		if x0 < 0 or x0 > image.shape[1]:
			return
		y0, y1 = 0, image.shape[0]
	elif math.isclose(cos_angle, 0.0):
		y0 = y1 = round(rho / sin_angle)
		if y0 < 0 or y0 > image.shape[0]:
			return
		x0, x1 = 0, image.shape[1]
	else:
		pts_on_boundary = list()
		x = 0
		y = round((rho - x * cos_angle) / sin_angle)
		if 0 <= y <= image.shape[0]:
			pts_on_boundary.append((x, y))
		x = image.shape[1]
		y = round((rho - x * cos_angle) / sin_angle)
		if 0 <= y <= image.shape[0] and (x, y) not in pts_on_boundary:
			pts_on_boundary.append((x, y))
		y = 0
		x = round((rho - y * sin_angle) / cos_angle)
		if 0 <= x <= image.shape[1] and (x, y) not in pts_on_boundary:
			pts_on_boundary.append((x, y))
		y = image.shape[0]
		x = round((rho - y * sin_angle) / cos_angle)
		if 0 <= x <= image.shape[1] and (x, y) not in pts_on_boundary:
			pts_on_boundary.append((x, y))
		assert len(pts_on_boundary) == 2, 'Points on boundary = {}'.format(pts_on_boundary)
		(x0, y0), (x1, y1) = pts_on_boundary[:2]

	cv2.line(image, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)

def detect_line():
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
	# TODO [check] >> OpenCV implementation of Hough line transform may have some bugs.
	#lines = cv2.HoughLines(edges, rho=1, theta=math.pi / 180, threshold=150, srn=0, stn=0, min_theta=0, max_theta=math.pi)  # The shape of lines = (#lines, 1, 2).
	lines = cv2.HoughLines(edges, rho=1, theta=math.pi / 180, threshold=150, srn=0, stn=0, min_theta=-math.pi / 2, max_theta=math.pi / 2)  # The shape of lines = (#lines, 1, 2).

	if lines is not None:
		print('#detected lines (HoughLines) = {}.'.format(len(lines)))
		offset = 1000
		for idx, line in enumerate(lines):
			# NOTE [info] >> Rho can be negative.
			#	When rho is negative, (rho, theta) = (-rho, theta - math.pi). -math.pi <= theta <= math.pi.
			#	https://docs.opencv.org/master/d6/d10/tutorial_py_houghlines.html
			rho, theta = line[0]
			rho, theta = float(rho), float(theta)
			#rho, theta = (float(rho), float(theta)) if rho >= 0 else (-float(rho), float(theta) - math.pi)

			#print('\t#{}: rho = {}, theta = {}.'.format(idx, rho, math.degrees(theta)))

			if True:
				draw_hough_line(img, rho, theta, (0, 0, 255), 1)
			else:
				cos_theta, sin_theta = math.cos(theta), math.sin(theta)
				x0, y0 = rho * cos_theta, rho * sin_theta
				dx, dy = offset * sin_theta, offset * cos_theta
				pt1 = (round(x0 - dx), round(y0 + dy))
				pt2 = (round(x0 + dx), round(y0 - dy))

				cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
	else:
		print('No detected line (HoughLines).')

	#--------------------
	#lines = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=10, minLineLength=500, maxLineGap=5)  # The shape of lines = (#lines, 1, 4).
	lines = cv2.HoughLinesP(edges, rho=1, theta=math.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)  # The shape of lines = (#lines, 1, 4).

	if lines is not None:
		print('#detected lines (HoughLinesP) = {}.'.format(len(lines)))
		for idx, line in enumerate(lines):
			x1, y1, x2, y2 = line[0]
			x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

			#print('\t#{}: ({}, {}) - ({}, {}).'.format(idx, x1, y1, x2, y2))

			cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
	else:
		print('No detected line (HoughLinesP).')

	cv2.imshow('Hough Line Transform', img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

def detect_circle():
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
		print('#detected circles = {}.'.format(len(circles[0])))
		circles = np.around(circles).astype(np.uint16)
		for circle in circles[0]:
			cx, cy, radius = circle[0], circle[1], circle[2]
			cx, cy, radius = int(cx), int(cy), int(radius)

			# Draw the circle center.
			cv2.circle(img, (cx, cy), 1, (0, 255, 0), cv2.FILLED, cv2.LINE_AA, 0)
			# Draw the circle outline.
			cv2.circle(img, (cx, cy), radius, (0, 0, 255), 1, cv2.LINE_AA, 0)
	else:
		print('No detected circle.')

	cv2.imshow('Hough Circle Transform', img)
	cv2.waitKey(0)

	cv2.destroyAllWindows()

# REF [site] >> https://docs.opencv.org/4.1.2/dd/d1a/group__imgproc__feature.html
def detect_line_using_point_set():
	points = np.array([
		[ 0,   369 ], [ 10,  364 ], [ 20,  358 ], [ 30,  352 ],
		[ 40,  346 ], [ 50,  341 ], [ 60,  335 ], [ 70,  329 ],
		[ 80,  323 ], [ 90,  318 ], [ 100, 312 ], [ 110, 306 ],
		[ 120, 300 ], [ 130, 295 ], [ 140, 289 ], [ 150, 284 ],
		[ 160, 277 ], [ 170, 271 ], [ 180, 266 ], [ 190, 260 ]
	], dtype=np.float32)
	points = np.expand_dims(points, axis=1)

	lines_max = 10
	threshold = 5
	# TODO [check] >> I guess that rho has relations with image size.
	rhoMin, rhoMax, rhoStep = 0, np.max(points), 1
	#thetaMin, thetaMax, thetaStep = 0, math.pi / 2, math.pi / 180
	thetaMin, thetaMax, thetaStep = -math.pi / 2, math.pi / 2, math.pi / 180
	# TODO [check] >> OpenCV implementation of Hough line transform may have some bugs.
	lines = cv2.HoughLinesPointSet(points, lines_max, threshold, rhoMin, rhoMax, rhoStep, thetaMin, thetaMax, thetaStep)

	if lines is not None:
		print('#detected lines = {}.'.format(len(lines)))

		img = np.zeros((500, 300, 3), dtype=np.uint8)
		offset = 1000
		for idx, line in enumerate(lines):
			# NOTE [info] >> Rho can be negative.
			#	When rho is negative, (rho, theta) = (-rho, theta - math.pi). -math.pi <= theta <= math.pi.
			#	https://docs.opencv.org/master/d6/d10/tutorial_py_houghlines.html
			votes, rho, theta = line[0]
			votes, rho, theta = int(votes), float(rho), float(theta)
			#votes, rho, theta = (int(votes), float(rho), float(theta)) if rho >= 0 else (int(votes), -float(rho), float(theta) - math.pi)

			print('\t#{}: votes = {}, rho = {}, theta = {}.'.format(idx, votes, rho, theta))

			if True:
				draw_hough_line(img, rho, theta, (0, 0, 255), 1)
			else:
				cos_theta, sin_theta = math.cos(theta), math.sin(theta)
				x0, y0 = rho * cos_theta, rho * sin_theta
				dx, dy = offset * sin_theta, offset * cos_theta
				pt1 = (round(x0 - dx), round(y0 + dy))
				pt2 = (round(x0 + dx), round(y0 - dy))

				cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
		for pt in points:
			cv2.circle(img, tuple(pt[0]), 1, (255, 0, 0), cv2.FILLED, cv2.LINE_AA)

		cv2.imshow('Hough Line Transform', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print('No detected line.')

def main():
	# TODO [check] >> OpenCV implementation of Hough line transform may have some bugs.
	#	Use skimage.transform.hough_line() & skimage.transform.hough_line_peaks() instead.

	#detect_line()
	#detect_circle()

	detect_line_using_point_set()

	# Hough space.
	# REF [file] >> ${SWDT_CPP_HOME}/rnd/test/machine_vision/opencv/opencv_hough_transform.cpp

	# Operation in Hough space.
	# REF [function] >> estimate_page_orientation_based_on_fft() in ${DataAnalysis_HOME}/app/document/estimate_document_orientation.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
