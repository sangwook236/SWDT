#!/usr/bin/env python

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
			x, y, w, h = cv2.boundingRect(contour)
			cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 2)

			# Rotated rectangle (OBB).
			rect = cv2.minAreaRect(contour)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			cv2.drawContours(rgb, [box], 0, (0, 0, 255), 2)

			# Minimum enclosing circle.
			(x, y), radius = cv2.minEnclosingCircle(contour)
			center = (int(x), int(y))
			radius = int(radius)
			cv2.circle(rgb, center, radius, (0, 255, 255), 2)

			# Fits an ellipse.
			ellipse = cv2.fitEllipse(contour)
			#ellipse = cv2.fitEllipseAMS(contour)
			#ellipse = cv2.fitEllipseDirect(contour)
			cv2.ellipse(rgb, ellipse, (255, 255, 0), 2)

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

def main():
	contour_feature()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
