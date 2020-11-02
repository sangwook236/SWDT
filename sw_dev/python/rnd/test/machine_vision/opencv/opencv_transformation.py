#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# REF [site] >> https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
def geometric_transformation():
	#--------------------
	# Scaling.

	img = cv.imread('../../../data/machine_vision/messi5.jpg', cv.IMREAD_COLOR)
	height, width = img.shape[:2]

	res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
	res = cv.resize(img, (2 * width, 2 * height), interpolation=cv.INTER_CUBIC)

	#--------------------
	# Translation.

	img = cv.imread('../../../data/machine_vision/messi5.jpg', cv.IMREAD_GRAYSCALE)
	rows, cols = img.shape

	M = np.float32([[1, 0, 100], [0, 1, 50]])
	dst = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_LINEAR)

	cv.imshow('img', dst)
	cv.waitKey(0)
	cv.destroyAllWindows()

	#--------------------
	# Rotation.

	img = cv.imread('../../../data/machine_vision/messi5.jpg', cv.IMREAD_GRAYSCALE)
	rows, cols = img.shape

	# cols-1 and rows-1 are the coordinate limits.
	M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 90, scale=1)
	dst = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_LINEAR)

	#--------------------
	# Affine transformation.

	img = cv.imread('../../../data/machine_vision/drawing.png', cv.IMREAD_COLOR)
	rows, cols, ch = img.shape

	pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
	pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

	M = cv.getAffineTransform(pts1, pts2)
	dst = cv.warpAffine(img, M, (cols, rows), flags=cv.INTER_LINEAR)

	plt.subplot(121), plt.imshow(img), plt.title('Input')
	plt.subplot(122), plt.imshow(dst), plt.title('Output')
	plt.show()

	#--------------------
	# Perspective transformation.

	img = cv.imread('../../../data/machine_vision/sudoku.png', cv.IMREAD_COLOR)
	rows, cols, ch = img.shape

	pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
	pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

	M = cv.getPerspectiveTransform(pts1, pts2, solveMethod=cv.DECOMP_LU)
	dst = cv.warpPerspective(img, M, (300, 300), flags=cv.INTER_LINEAR)

	plt.subplot(121), plt.imshow(img), plt.title('Input')
	plt.subplot(122), plt.imshow(dst), plt.title('Output')
	plt.show()

def ellipse_transformation_example():
	#for rangle in range(0, 360, 10):
	for rangle in range(-90, 90, 10):
		rgb = np.zeros((500, 500, 3), np.uint8)
		cv.ellipse(rgb, (300, 200), (100, 50), rangle, 0, 360, (127, 127, 127), cv.FILLED, cv.LINE_8)

		pts = np.stack(np.nonzero(rgb)).transpose()[:,[1, 0]]

		# Rotated rectangle (OBB).
		obb = cv.minAreaRect(pts)  # Tuple: (center, size, angle).
		obb_center, obb_size, obb_angle = obb
		obb_pts = cv.boxPoints(obb)  # 4 x 2. np.float32.
		#obb_pts = np.int0(obb_pts)

		# Straight bounding rectangle (AABB).
		aabb = cv.boundingRect(pts)  # x (left), y (top), width, height.
		aabb_x1, aabb_y1, aabb_x2, aabb_y2 = aabb[0], aabb[1], aabb[0] + aabb[2], aabb[1] + aabb[3]

		# The largest bbox.
		(left, top), (right, bottom) = np.min(obb_pts, axis=0), np.max(obb_pts, axis=0)

		cv.rectangle(rgb, (left, top), (right, bottom), (255, 255, 0), 2, cv.LINE_8)
		cv.rectangle(rgb, (aabb_x1, aabb_y1), (aabb_x2, aabb_y2), (0, 255, 255), 2, cv.LINE_8)
		if False:
			cv.drawContours(rgb, [np.int0(obb_pts)], 0, (0, 0, 255), 2, cv.LINE_8)
		else:
			cv.line(rgb, (math.floor(obb_pts[1,0]), math.floor(obb_pts[1,1])), (math.floor(obb_pts[2,0]), math.floor(obb_pts[2,1])), (0, 0, 255), 2, cv.LINE_8)
			cv.line(rgb, (math.floor(obb_pts[2,0]), math.floor(obb_pts[2,1])), (math.floor(obb_pts[3,0]), math.floor(obb_pts[3,1])), (0, 255, 0), 2, cv.LINE_8)
			cv.line(rgb, (math.floor(obb_pts[3,0]), math.floor(obb_pts[3,1])), (math.floor(obb_pts[0,0]), math.floor(obb_pts[0,1])), (255, 0, 0), 2, cv.LINE_8)
			cv.line(rgb, (math.floor(obb_pts[0,0]), math.floor(obb_pts[0,1])), (math.floor(obb_pts[1,0]), math.floor(obb_pts[1,1])), (255, 0, 255), 2, cv.LINE_8)

		cv.imshow('Input Image', rgb)

		#--------------------
		# Rotation.

		rot_size, rot_angle = obb_size, obb_angle
		# TODO [check] >>
		if obb_size[0] >= obb_size[1]:
		#if obb_angle < -10 or obb_angle > 10:
			rot_size, rot_angle = obb_size[1::-1], obb_angle + 90

		if False:
			# Too big.
			dia = math.ceil(math.sqrt(rgb.shape[0]**2 + rgb.shape[1]**2))
			R = cv.getRotationMatrix2D(obb_center, angle=rot_angle, scale=1)
			rotated = cv.warpAffine(rgb, R, (dia, dia), flags=cv.INTER_LINEAR)

			rotated_patch = rotated[math.floor(obb_center[1] - rot_size[1] / 2):math.ceil(obb_center[1] + rot_size[1] / 2), math.floor(obb_center[0] - rot_size[0] / 2):math.ceil(obb_center[0] + rot_size[0] / 2)]
		elif False:
			# Trimmed.
			patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1) - 1), max(0, math.floor(aabb_y1) - 1), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
			#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1)), max(0, math.floor(aabb_y1)), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
			patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

			ctr = patch.shape[1] / 2, patch.shape[0] / 2
			dia = math.ceil(math.sqrt(patch.shape[0]**2 + patch.shape[1]**2))
			R = cv.getRotationMatrix2D(ctr, angle=rot_angle, scale=1)
			rotated = cv.warpAffine(patch, R, (dia, dia), flags=cv.INTER_LINEAR)

			rotated_patch = rotated[:patch.shape[0],:patch.shape[1]]
		else:
			radius = math.sqrt(rot_size[0]**2 + rot_size[1]**2) / 2
			dia = math.ceil(radius * 2)

			patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius) - 1), max(0, math.floor(obb_center[1] - radius) - 1), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
			#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius)), max(0, math.floor(obb_center[1] - radius)), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
			patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

			ctr = patch.shape[1] / 2, patch.shape[0] / 2
			R = cv.getRotationMatrix2D(ctr, angle=rot_angle, scale=1)
			rotated = cv.warpAffine(patch, R, (dia, dia), flags=cv.INTER_LINEAR)

			rotated_patch = rotated[math.floor(ctr[1] - rot_size[1] / 2):math.ceil(ctr[1] + rot_size[1] / 2), math.floor(ctr[0] - rot_size[0] / 2):math.ceil(ctr[0] + rot_size[0] / 2)]

		cv.imshow('Rotated Image (OBB)', rotated)
		cv.imshow('Rotated Image (OBB, Patched)', rotated_patch)

		patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1) - 1), max(0, math.floor(aabb_y1) - 1), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
		#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1)), max(0, math.floor(aabb_y1)), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
		patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

		ctr = patch.shape[1] / 2, patch.shape[0] / 2
		#ctr = tuple(np.array(patch.shape[1::-1]) / 2)
		R = cv.getRotationMatrix2D(ctr, angle=rot_angle, scale=1)
		rotated = cv.warpAffine(patch, R, (300, 150), flags=cv.INTER_LINEAR)

		cv.imshow('Rotated Image (AABB)', rotated)

		import scipy.ndimage
		rotated_scipy = scipy.ndimage.rotate(patch, rot_angle, reshape=True)
		ctr = rotated_scipy.shape[1] / 2, rotated_scipy.shape[0] / 2
		rotated_scipy_patch = rotated_scipy[math.floor(ctr[1] - rot_size[1] / 2):math.ceil(ctr[1] + rot_size[1] / 2), math.floor(ctr[0] - rot_size[0] / 2):math.ceil(ctr[0] + rot_size[0] / 2)]

		cv.imshow('Rotated Image (scipy)', rotated_scipy)
		cv.imshow('Rotated Image (scipy, Patch)', rotated_scipy_patch)

		#--------------------
		# Affine transformation.

		if False:
			if obb_size[0] >= obb_size[1]:
				target_pts = np.float32([[0, 100], [0, 0], [100, 0], [100, 100]])
			else:
				target_pts = np.float32([[100, 100], [0, 100], [0, 0], [100, 0]])
			canvas_size = 300, 150
		else:
			if obb_size[0] >= obb_size[1]:
				target_pts = np.float32([[0, obb_size[1]], [0, 0], [obb_size[0], 0], [obb_size[0], obb_size[1]]])
				canvas_size = round(obb_size[0]), round(obb_size[1])
			else:
				target_pts = np.float32([[obb_size[1], obb_size[0]], [0, obb_size[0]], [0, 0], [obb_size[1], 0]])
				canvas_size = round(obb_size[1]), round(obb_size[0])

		A = cv.getAffineTransform(obb_pts[:3], target_pts[:3])  # Three points.
		warped = cv.warpAffine(rgb, A, canvas_size, flags=cv.INTER_LINEAR)

		cv.imshow('Affinely Warped Image', warped)

		#--------------------
		# Perspective transformation.

		T = cv.getPerspectiveTransform(obb_pts, target_pts, solveMethod=cv.DECOMP_LU)  # Four points.
		warped = cv.warpPerspective(rgb, T, canvas_size, flags=cv.INTER_LINEAR)

		cv.imshow('Perspectively Warped Image', warped)

		cv.waitKey(0)
	cv.destroyAllWindows()

def measure_time():
	import scipy.ndimage

	rgb = np.zeros((500, 500, 3), np.uint8)
	cv.ellipse(rgb, (300, 200), (100, 50), 30, 0, 360, (255, 0, 255), cv.FILLED, cv.LINE_8)

	pts = np.stack(np.nonzero(rgb)).transpose()[:,[1, 0]]

	# Rotated rectangle (OBB).
	obb = cv.minAreaRect(pts)  # Tuple: (center, size, angle).
	obb_pts = cv.boxPoints(obb)  # 4 x 2. np.float32.
	#obb_pts = np.int0(obb_pts)

	obb_center, obb_size, obb_angle = obb
	#obb_size, obb_angle = obb_size[1::-1], obb_angle + 90

	import time
	num_iterations = 1000
	print('Start measuring rotation (opencv)...')
	start_time = time.time()
	for _ in range(num_iterations):
		radius = math.sqrt(obb_size[0]**2 + obb_size[1]**2) / 2
		dia = math.ceil(radius * 2)

		patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius) - 1), max(0, math.floor(obb_center[1] - radius) - 1), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
		#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(obb_center[0] - radius)), max(0, math.floor(obb_center[1] - radius)), min(rgb.shape[1], math.ceil(obb_center[0] + radius) + 1), min(rgb.shape[0], math.ceil(obb_center[1] + radius) + 1)
		patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

		ctr = patch.shape[1] / 2, patch.shape[0] / 2
		R = cv.getRotationMatrix2D(ctr, angle=obb_angle, scale=1)
		rotated = cv.warpAffine(patch, R, (dia, dia), flags=cv.INTER_LINEAR)

		rotated_patch = rotated[math.floor(ctr[1] - obb_size[1] / 2):math.ceil(ctr[1] + obb_size[1] / 2), math.floor(ctr[0] - obb_size[0] / 2):math.ceil(ctr[0] + obb_size[0] / 2)]
	print('End measuring rotation (opencv): {} secs.'.format(time.time() - start_time))

	print('Start measuring rotation (scipy)...')
	start_time = time.time()
	for _ in range(num_iterations):
		aabb = cv.boundingRect(pts)  # x (left), y (top), width, height.
		aabb_x1, aabb_y1, aabb_x2, aabb_y2 = aabb[0], aabb[1], aabb[0] + aabb[2], aabb[1] + aabb[3]

		patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1) - 1), max(0, math.floor(aabb_y1) - 1), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
		#patch_x1, patch_y1, patch_x2, patch_y2 = max(0, math.floor(aabb_x1)), max(0, math.floor(aabb_y1)), min(rgb.shape[1], math.ceil(aabb_x2) + 1), min(rgb.shape[0], math.ceil(aabb_y2) + 1)
		patch = rgb[patch_y1:patch_y2, patch_x1:patch_x2]

		rotated_scipy = scipy.ndimage.rotate(patch, obb_angle, reshape=True)
		ctr = rotated_scipy.shape[1] / 2, rotated_scipy.shape[0] / 2
		rotated_scipy_patch = rotated_scipy[math.floor(ctr[1] - obb_size[1] / 2):math.ceil(ctr[1] + obb_size[1] / 2), math.floor(ctr[0] - obb_size[0] / 2):math.ceil(ctr[0] + obb_size[0] / 2)]
	print('End measuring rotation (scipy): {} secs.'.format(time.time() - start_time))

	print('Start measuring affine transformation...')
	start_time = time.time()
	for _ in range(num_iterations):
		target_pts = np.float32([[0, obb_size[1]], [0, 0], [obb_size[0], 0]])
		A = cv.getAffineTransform(obb_pts[:3], target_pts)  # Three points.
		warped = cv.warpAffine(rgb, A, (round(obb_size[0]), round(obb_size[1])), flags=cv.INTER_LINEAR)
	print('End measuring affine transformation: {} secs.'.format(time.time() - start_time))

	print('Start measuring perspective transformation...')
	start_time = time.time()
	for _ in range(num_iterations):
		target_pts = np.float32([[0, obb_size[1]], [0, 0], [obb_size[0], 0], [obb_size[0], obb_size[1]]])
		T = cv.getPerspectiveTransform(obb_pts, target_pts, solveMethod=cv.DECOMP_LU)  # Four points.
		warped = cv.warpPerspective(rgb, T, (round(obb_size[0]), round(obb_size[1])), flags=cv.INTER_LINEAR)
	print('End measuring perspective transformation: {} secs.'.format(time.time() - start_time))

def main():
	#geometric_transformation()

	ellipse_transformation_example()
	#measure_time()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
