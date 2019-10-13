#!/usr/bin/env python

# REF [site] >> https://matplotlib.org/users/image_tutorial.html

#%matplotlib inline
#%matplotlib notebook

#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def basic_example():
	# Loading image data is supported by the Pillow library.
	# Natively, matplotlib only supports PNG images.

	img = mpimg.imread('./data/stinkbug.png')

	imgplot = plt.imshow(img)

	lum_img = img[:,:,0]
	plt.imshow(lum_img)
	plt.imshow(lum_img, cmap='hot')
	plt.imshow(lum_img, cmap='gray')

	imgplot = plt.imshow(lum_img)
	imgplot.set_cmap('nipy_spectral')

	imgplot = plt.imshow(lum_img)
	plt.colorbar()

	plt.imsave('./data/img_gray.png', lum_img, cmap='gray')

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
