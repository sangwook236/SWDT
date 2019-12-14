#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import skimage
import skimage.feature
import matplotlib.pyplot as plt
import matplotlib

#---------------------------------------------------------------------

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
def hough_line_transform_example():
	# Construct a test image.
	image = np.zeros((200, 200))
	idx = np.arange(25, 175)
	image[idx[::-1], idx] = 255
	image[idx, idx] = 255

	# Classic straight-line Hough transform.
	# Set a precision of 0.5 degree.
	tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
	hspace, theta, rho = skimage.transform.hough_line(image, theta=tested_angles)

	# Generate figure.
	fig, axes = plt.subplots(1, 3, figsize=(15, 6))
	ax = axes.ravel()

	ax[0].imshow(image, cmap=matplotlib.cm.gray)
	ax[0].set_title('Input image')
	ax[0].set_axis_off()

	ax[1].imshow(np.log(1 + hspace),
				 extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
				 cmap=matplotlib.cm.gray, aspect=1 / 1.5)
	ax[1].set_title('Hough transform')
	ax[1].set_xlabel('Angles (degrees)')
	ax[1].set_ylabel('Distance (pixels)')
	ax[1].axis('image')

	ax[2].imshow(image, cmap=matplotlib.cm.gray)
	origin = np.array((0, image.shape[1]))
	for _, angle, dist in zip(*skimage.transform.hough_line_peaks(hspace, theta, rho)):
		y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
		ax[2].plot(origin, (y0, y1), '-r')
	ax[2].set_xlim(origin)
	ax[2].set_ylim((image.shape[0], 0))
	ax[2].set_axis_off()
	ax[2].set_title('Detected lines')

	plt.tight_layout()
	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
def probabilistic_hough_line_transform_example():
	# Line finding using the Probabilistic Hough Transform.
	image = skimage.data.camera()
	edges = skimage.feature.canny(image, 2, 1, 25)
	lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)

	# Generate figure.
	fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(image, cmap=matplotlib.cm.gray)
	ax[0].set_title('Input image')

	ax[1].imshow(edges, cmap=matplotlib.cm.gray)
	ax[1].set_title('Canny edges')

	ax[2].imshow(edges * 0)
	for line in lines:
		p0, p1 = line
		ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
	ax[2].set_xlim((0, image.shape[1]))
	ax[2].set_ylim((image.shape[0], 0))
	ax[2].set_title('Probabilistic Hough')

	for a in ax:
		a.set_axis_off()

	plt.tight_layout()
	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py
def hough_circle_transform_example():
	# Load picture and detect edges.
	image = skimage.img_as_ubyte(skimage.data.coins()[160:230, 70:270])
	edges = skimage.feature.canny(image, sigma=3, low_threshold=10, high_threshold=50)

	# Detect two radii.
	hough_radii = np.arange(20, 35, 2)
	hough_res = skimage.transform.hough_circle(edges, hough_radii)

	# Select the most prominent 3 circles.
	accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

	# Draw them.
	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
	image = skimage.color.gray2rgb(image)
	for center_y, center_x, radius in zip(cy, cx, radii):
		circy, circx = skimage.draw.circle_perimeter(center_y, center_x, radius, shape=image.shape)
		image[circy, circx] = (220, 20, 20)

	ax.imshow(image, cmap=plt.cm.gray)
	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html#sphx-glr-auto-examples-edges-plot-circular-elliptical-hough-transform-py
def hough_ellipse_transform_example():
	# Load picture, convert to grayscale and detect edges.
	image_rgb = skimage.data.coffee()[0:220, 160:420]
	image_gray = skimage.color.rgb2gray(image_rgb)
	edges = skimage.feature.canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

	# Perform a Hough Transform.
	# The accuracy corresponds to the bin size of a major axis.
	# The value is chosen in order to get a single high accumulator.
	# The threshold eliminates low accumulators.
	result = skimage.transform.hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
	result.sort(order='accumulator')

	# Estimated parameters for the ellipse.
	best = list(result[-1])
	yc, xc, a, b = [int(round(x)) for x in best[1:5]]
	orientation = best[5]

	# Draw the ellipse on the original image.
	cy, cx = skimage.draw.ellipse_perimeter(yc, xc, a, b, orientation)
	image_rgb[cy, cx] = (0, 0, 255)
	# Draw the edge (white) and the resulting ellipse (red).
	edges = skimage.color.gray2rgb(skimage. img_as_ubyte(edges))
	edges[cy, cx] = (250, 0, 0)

	fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)

	ax1.set_title('Original picture')
	ax1.imshow(image_rgb)

	ax2.set_title('Edge (white) and result (red)')
	ax2.imshow(edges)

	plt.show()

def main():
	#hough_line_transform_example()
	#probabilistic_hough_line_transform_example()
	hough_circle_transform_example()
	hough_ellipse_transform_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
