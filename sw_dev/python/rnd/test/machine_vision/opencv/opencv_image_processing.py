#!/usr/bin/env python

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# REF [site] >> https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
def thresholding():
	img = cv.imread('gradient.png', 0)

	ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
	ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
	ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
	ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

	titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
	images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

	for i in xrange(6):
		plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	#--------------------
	img = cv.imread('sudoku.png', 0)
	img = cv.medianBlur(img, 5)

	ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
	th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
	th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

	titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
	images = [img, th1, th2, th3]

	for i in xrange(4):
		plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])
	plt.show()

	#--------------------
	img = cv.imread('noisy2.png', 0)

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

	for i in xrange(3):
		plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
		plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
		plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
		plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
		plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
		plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
	plt.show()

# REF [site] >> https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
def histogram_equalization():
	img = cv.imread('wiki.jpg', 0)

	hist, bins = np.histogram(img.flatten(), 256, [0, 256])

	cdf = hist.cumsum()
	cdf_normalized = cdf * float(hist.max()) / cdf.max()

	plt.plot(cdf_normalized, color='b')
	plt.hist(img.flatten(), 256, [0, 256], color='r')
	plt.xlim([0, 256])
	plt.legend(('cdf', 'histogram'), loc='upper left')
	plt.show()

	#--------------------
	img = cv.imread('wiki.jpg', 0)
	equ = cv.equalizeHist(img)
	res = np.hstack((img, equ))  # Stacking images side-by-side.
	cv.imwrite('res.png', res)

	#--------------------
	# Contrast Limited Adaptive Histogram Equalization (CLAHE).

	img = cv.imread('tsukuba_l.png', 0)

	# Create a CLAHE object (Arguments are optional).
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl1 = clahe.apply(img)

	cv.imwrite('clahe_2.jpg', cl1)

# REF [site] >> https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
def geometric_transformation():
	#--------------------
	# Scaling.

	img = cv.imread('messi5.jpg')
	height, width = img.shape[:2]

	res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
	res = cv.resize(img, (2*width, 2*height), interpolation=cv.INTER_CUBIC)

	#--------------------
	# Translation.

	img = cv.imread('messi5.jpg', 0)
	rows, cols = img.shape

	M = np.float32([[1, 0, 100], [0, 1, 50]])
	dst = cv.warpAffine(img, M, (cols, rows))

	cv.imshow('img', dst)
	cv.waitKey(0)
	cv.destroyAllWindows()

	#--------------------
	# Rotation.

	img = cv.imread('messi5.jpg', 0)
	rows, cols = img.shape

	# cols-1 and rows-1 are the coordinate limits.
	M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
	dst = cv.warpAffine(img, M, (cols, rows))

	#--------------------
	# Affine transformation.

	img = cv.imread('drawing.png')
	rows, cols, ch = img.shape

	pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
	pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

	M = cv.getAffineTransform(pts1, pts2)
	dst = cv.warpAffine(img, M, (cols, rows))

	plt.subplot(121), plt.imshow(img), plt.title('Input')
	plt.subplot(122), plt.imshow(dst), plt.title('Output')
	plt.show()

	#--------------------
	# Perspective transformation.

	img = cv.imread('sudoku.png')
	rows, cols, ch = img.shape

	pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
	pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

	M = cv.getPerspectiveTransform(pts1, pts2)
	dst = cv.warpPerspective(img, M, (300, 300))

	plt.subplot(121), plt.imshow(img), plt.title('Input')
	plt.subplot(122), plt.imshow(dst), plt.title('Output')
	plt.show()

def main():
	# REF [site] >> https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html

	thresholding()
	histogram_equalization()

	geometric_transformation()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
