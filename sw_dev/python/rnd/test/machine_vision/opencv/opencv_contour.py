#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2

# REF [site] >> https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
def contour_feature():
	image_filepaths = [
		'../../../data/machine_vision/lightning.png',
		'../../../data/machine_vision/boot.png',
	]

	for image_filepath in image_filepaths:
		rgb = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
		if rgb is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue

		gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

		# Find contours in the thresholded image, then initialize the list of digit locations.
		#img2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		_, contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		if False:
			contours_filtered = list()
			for contour in contours:
				if 500 <= cv2.contourArea(contour) <= 10000:
					# Fits an ellipse.
					ellipse = cv2.fitEllipse(contour)
					#ellipse = cv2.fitEllipseAMS(contour)
					#ellipse = cv2.fitEllipseDirect(contour)
					if ellipse[1][0] > 4 and ellipse[1][1] / ellipse[1][0] >= 5:
						contours_filtered.append(contour)
		else:
			contours_filtered = contours
		cv2.drawContours(rgb, contours_filtered, -1, (0, 255, 0), 3, cv2.LINE_AA)

		for contour in contours_filtered:
			hull = cv2.convexHull(contour)
			cv2.drawContours(rgb, [hull], 0, (127, 127, 127), 2)
			print('Is convex?', cv2.isContourConvex(contour), cv2.isContourConvex(hull))

			# Straight bounding rectangle (AABB).
			x, y, w, h = cv2.boundingRect(contour)  # x (left), y (top), width, height.
			cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_8)

			# Rotated rectangle (OBB).
			obb = cv2.minAreaRect(contour)  # Tuple: (center, size, angle).
			obb_pts = cv2.boxPoints(obb)  # 4 x 2. np.float32.
			#obb_pts = np.int0(obb_pts)
			obb_pts = np.round(obb_pts).astype(np.int)
			cv2.drawContours(rgb, [obb_pts], 0, (0, 0, 255), 2, cv2.LINE_8)

			# Minimum enclosing circle.
			(x, y), radius = cv2.minEnclosingCircle(contour)
			center = (int(x), int(y))
			radius = int(radius)
			cv2.circle(rgb, center, radius, (0, 255, 255), 2, cv2.LINE_8)

			# Fits an ellipse.
			ellipse = cv2.fitEllipse(contour)
			#ellipse = cv2.fitEllipseAMS(contour)
			#ellipse = cv2.fitEllipseDirect(contour)
			cv2.ellipse(rgb, ellipse, (255, 255, 0), 2, cv2.LINE_8)

			# Fits a line.
			if False:
				# Uses points on contour.
				[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
			elif True:
				# Uses all points in contour.
				mask = np.zeros_like(gray)
				cv2.drawContours(mask, [contour], 0, (255, 255, 255), cv2.FILLED)
				idx = cv2.findNonZero(mask)
				[vx, vy, x, y] = cv2.fitLine(idx, cv2.DIST_L2, 0, 0.01, 0.01)
			else:
				# Uses non-zero points in contour.
				mask = np.zeros_like(gray)
				cv2.drawContours(mask, [contour], 0, (255, 255, 255), cv2.FILLED)
				mask = np.where(mask > 0, gray, mask)
				idx = cv2.findNonZero(mask)
				[vx, vy, x, y] = cv2.fitLine(idx, cv2.DIST_L2, 0, 0.01, 0.01)
			rows, cols = rgb.shape[:2]
			lefty = int((-x * vy / vx) + y)
			righty = int(((cols - x) * vy / vx) + y)
			cv2.line(rgb, (cols - 1, righty), (0, lefty), (255, 0, 255), 2)

			# PCA.
			# REF [file] >> ./opencv_pca.py
			
			# Find the largest rectangle inscribed in a non-convex polygon.
			# REF [site] >> http://d3plus.org/blog/behind-the-scenes/2014/07/08/largest-rect/
			# REF [site] >> https://stackoverflow.com/questions/32674256/how-to-adapt-or-resize-a-rectangle-inside-an-object-without-including-or-with-a?lq=1

		cv2.imshow('Contour', rgb)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

def minAreaRect_test():
	for angle in range(-360, 370, 10):
		pts = cv2.boxPoints(((200, 200), (200, 40), angle))
		num_pts = len(pts)

		# The coordinate system.
		#	X: rightward, Y: downward, CW.
		# If obb_size[0] > obb_size[1], obb_angle = the angle between the +X axis and the major axis in quadrant I. [deg].
		# If obb_size[0] < obb_size[1], obb_angle = the angle between the +X axis and the minor axis in quadrant I. [deg].
		obb_center, obb_size, obb_angle = cv2.minAreaRect(pts)
		assert 0 <= obb_angle <= 90, '0 <= {} <= 90'.format(obb_angle)  # 0 <= obb_angle <= 90. (?)
		#obb_angle = math.floor((90 - obb_angle) / 180)  # -90 <= obb_angle <= 90.
		#obb_angle = math.ceil((-90 - obb_angle) / 180)  # -90 <= obb_angle <= 90.
		#assert -90 <= obb_angle <= 90, '-90 <= {} <= 90'.format(obb_angle)

		print("{}: OBB's size = {}, OBB's angle = {}.".format(angle, obb_size, obb_angle))

		img = np.zeros((400, 400, 3), dtype=np.uint8)
		cv2.drawContours(img, [np.int0(pts)], 0, (255, 255, 255), 2)
		colors = [
			(0, 0, 255),
			(0, 255, 0),
			(255, 0, 0),
			(255, 0, 255),
		]
		for idx in range(num_pts):
			cv2.circle(img, tuple(pts[idx].astype(np.int)), 4, colors[idx], cv2.FILLED)
			#cv2.line(img, tuple(pts[idx].astype(np.int)), tuple(pts[(idx + 1) % num_pts].astype(np.int)), colors[idx], 2)

		cv2.imshow('Image', img)
		cv2.waitKey(0)

def main():
	contour_feature()

	#minAreaRect_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
