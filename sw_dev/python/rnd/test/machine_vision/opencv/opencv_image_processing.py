#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# REF [site] >> https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
def thresholding():
	img = cv.imread('../../../data/machine_vision/gradient.png', cv.IMREAD_GRAYSCALE)

	ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
	ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
	ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
	ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

	titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

	for i in range(6):
		plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	#--------------------
	img = cv.imread('../../../data/machine_vision/sudoku.png', cv.IMREAD_GRAYSCALE)
	img = cv.medianBlur(img, 5)

	ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
	th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

	titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in range(4):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	#--------------------
	img = cv.imread('../../../data/machine_vision/noisy2.png', cv.IMREAD_GRAYSCALE)

	# Global thresholding.
	ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	# Otsu's thresholding.
	ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
	# Otsu's thresholding after Gaussian filtering.
	blur = cv.GaussianBlur(img, (5, 5), 0)
	ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

	# Plot all the images and their histograms.
	images = [img, 0, th1,
			  img, 0, th2,
			  blur, 0, th3]
	titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
			  'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
			  'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

	for i in range(3):
		plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
		plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
		plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
		plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
		plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
		plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
	plt.show()

# REF [site] >> https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization():
	img = cv.imread('../../../data/machine_vision/wiki.jpg', cv.IMREAD_GRAYSCALE)

	hist, bins = np.histogram(img.flatten(), 256, [0, 256])

	cdf = hist.cumsum()
	cdf_normalized = cdf * float(hist.max()) / cdf.max()

	plt.plot(cdf_normalized, color='b')
	plt.hist(img.flatten(), 256, [0, 256], color='r')
	plt.xlim([0, 256])
	plt.legend(('cdf', 'histogram'), loc='upper left')
	plt.show()

	#--------------------
	img = cv.imread('../../../data/machine_vision/wiki.jpg', cv.IMREAD_GRAYSCALE)

	equ = cv.equalizeHist(img)
	res = np.hstack((img, equ))  # Stacking images side-by-side.

	#cv.imwrite('./res.png', res)
	cv.imshow('Histogram Equalization', res)
	cv.waitKey(0)

	#--------------------
	# Contrast Limited Adaptive Histogram Equalization (CLAHE).

	img = cv.imread('../../../data/machine_vision/tsukuba_l.png', cv.IMREAD_GRAYSCALE)  # (height, width).

	# Create a CLAHE object (Arguments are optional).
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl1 = clahe.apply(img)

	#cv.imwrite('./clahe_2.jpg', cl1)
	cv.imshow('CLAHE', cl1)
	cv.waitKey(0)

def edge_detection():
	def draw_line(edge, gray):
		lines = cv.HoughLines(edge, rho=1, theta=math.pi / 180, threshold=100, srn=0, stn=0)

		rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
		if lines is not None:
			for line in lines[:100]:
				rho, theta = line[0]

				a, b = math.cos(theta), math.sin(theta)
				x0, y0 = a * rho, b * rho
				pt1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * (a))))
				pt2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * (a))))
				cv.line(rgb, pt1, pt2, (255, 0, 0), 2, cv.LINE_AA)

		cv.imshow('HoughLines', rgb)

		#--------------------
		lines = cv.HoughLinesP(edge, rho=1, theta=math.pi / 180, threshold=50, minLineLength=20, maxLineGap=20)

		rgb = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]
				cv.line(rgb, (x1, y1), (x2, y2), (255, 0, 0), 2, cv.LINE_AA)

		cv.imshow('HoughLinesP', rgb)

	if False:
		image_filepath = '../../../data/machine_vision/build.png'

		img = cv.imread(image_filepath, cv.IMREAD_GRAYSCALE)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			return

		img = cv.pyrUp(cv.pyrDown(img))
		#img = cv.GaussianBlur(img, (5, 5), 0)
		#img = cv.medianBlur(img, ksize=5)
		#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		#img = clahe.apply(img)

		dx = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
		dy = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=3)

		grad = cv.addWeighted(np.absolute(dx), 0.5, np.absolute(dy), 0.5, 0)

		_, grad = cv.threshold(grad, 100, 255, cv.THRESH_BINARY)

		cv.imshow('Image', img)
		cv.imshow('Gradient', grad)
		draw_line(((grad / np.max(grad)) * 255).astype(np.uint8), img)
		cv.waitKey(0)

	if False:
		image_filepath = '../../../data/machine_vision/build.png'

		img = cv.imread(image_filepath, cv.IMREAD_COLOR)
		if img is None:
			print('Failed to load an image: {}.'.format(image_filepath))
			return

		# Laplacian on gray scale.
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		gray = cv.pyrUp(cv.pyrDown(gray))
		#gray = cv.GaussianBlur(gray, (5, 5), 0)
		#gray = cv.medianBlur(gray, ksize=5)
		#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		#gray = clahe.apply(gray)

		dst = cv.Laplacian(gray, ddepth=-1, ksize=3)

		#_, dst = cv.threshold(dst, 50, 255, cv.THRESH_BINARY)

		cv.imshow('Laplacian - Grayscale', dst)
		draw_line(dst, gray)

		# Laplacian on color.
		planes = cv.split(img)
		planes_laplacian = list()
		for plane in planes:
			plane = cv.Laplacian(plane, ddepth=-1, ksize=3)
			plane = cv.convertScaleAbs(plane, alpha=1, beta=0)
			planes_laplacian.append(plane)

		dst = cv.merge(planes_laplacian)

		cv.imshow('Laplacian - Color', dst)
		#draw_line(cv.cvtColor(dst, cv.COLOR_BGR2GRAY), gray)
		cv.waitKey(0)

	if False:
		image_filepath = '../../../data/machine_vision/build.png'

		img = cv.imread(image_filepath, cv.IMREAD_GRAYSCALE)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			return

		#img = cv.pyrUp(cv.pyrDown(img))
		#img = cv.GaussianBlur(img, (5, 5), 0)
		#img = cv.medianBlur(img, ksize=5)
		#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		#img = clahe.apply(img)

		morphed = cv.morphologyEx(img, cv.MORPH_GRADIENT, None, None)  # Apply a dilate - Erode.

		_, morphed = cv.threshold(morphed, 30, 255, cv.THRESH_BINARY_INV)

		cv.imshow('Image', img)
		cv.imshow('Morphed', morphed)
		draw_line(morphed, img)
		cv.waitKey(0)

	if True:
		image_filepath = '../../../data/machine_vision/road.png'

		img = cv.imread(image_filepath, cv.IMREAD_GRAYSCALE)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			return

		#img = cv.pyrUp(cv.pyrDown(img))
		#img = cv.GaussianBlur(img, (5, 5), 0)
		#img = cv.medianBlur(img, ksize=5)
		#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
		#img = clahe.apply(img)

		dst = cv.Canny(img, 200, 200, apertureSize=3)
		_, dst = cv.threshold(dst, 100, 255, cv.THRESH_BINARY)

		cv.imshow('Image', img)
		cv.imshow('Canny', dst)
		draw_line(dst, img)
		cv.waitKey(0)

	cv.destroyAllWindows()

# REF [site] >> http://www.robindavid.fr/opencv-tutorial/chapter5-line-edge-and-contours-detection.html
def corner_detection():
	image_filepath = '../../../data/machine_vision/build.png'

	img = cv.imread(image_filepath, cv.IMREAD_GRAYSCALE)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return

	dst_32f = cv.cornerHarris(img, blockSize=3, ksize=3, k=0.01)

	dilated = cv.dilate(dst_32f, kernel=None, anchor=None, iterations=1)
	localMax = cv.compare(dst_32f, dilated, cv.CMP_EQ)  # Binary image.

	minv, maxv, minl, maxl = cv.minMaxLoc(dst_32f)
	_, dst_32f = cv.threshold(dst_32f, 0.01 * maxv, 255, cv.THRESH_BINARY)

	cornerMap = (dst_32f * 255).astype(np.int32)
	cornerMap[localMax == 0] = 0  # Deletes all modified pixels.

	centers = np.where(cornerMap > 0)
	#centers = np.vstack(centers.tolist()).T
	centers = list(zip(centers[1].tolist(), centers[0].tolist()))

	for center in centers:
		cv.circle(img, center, 3, (255, 255, 255), 2, cv.LINE_AA)

	cv.imshow('Image', img)
	cv.imshow('CornerHarris Result', dst_32f)
	cv.imshow('Unique Points after Dilatation/CMP/And', cornerMap)
	cv.waitKey(0)

	cv.destroyAllWindows()

def contour_detection():
	image_filepath = '../../../data/machine_vision/build.png'

	img = cv.imread(image_filepath, cv.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return

	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	_, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY)
	cv.imshow('Threshold 1', gray)

	element = cv.getStructuringElement(cv.MORPH_RECT, (5 * 2 + 1, 5 * 2 + 1), (5, 5))
	gray = cv.morphologyEx(gray, cv.MORPH_OPEN, element)
	gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, element)
	_, gray = cv.threshold(gray, 128, 255, cv.THRESH_BINARY_INV)

	cv.imshow('After MorphologyEx', gray)

	#--------------------
	_, contours, _ = cv.findContours(gray, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

	cv.drawContours(img, contours, -1, (0, 0, 255), 2, cv.FILLED)

	cv.imshow('Image', img)
	cv.waitKey(0)

	cv.destroyAllWindows()

def main():
	# REF [site] >> https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html

	#thresholding()
	#histogram_equalization()

	edge_detection()
	#corner_detection()
	#contour_detection()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
