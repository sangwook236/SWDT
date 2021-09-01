#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://matplotlib.org/users/image_tutorial.html

#import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline
#%matplotlib notebook

def basic_example():
	# Loading image data is supported by the Pillow library.
	# Natively, matplotlib only supports PNG images.

	image_filepath = './data/stinkbug.png'

	img = mpimg.imread(image_filepath)
	#img = plt.imread(image_filepath, format='png')

	imgplot = plt.imshow(img)

	lum_img = img[:,:,0]
	plt.imshow(lum_img)
	plt.imshow(lum_img, cmap='hot')
	plt.imshow(lum_img, cmap='gray')

	imgplot = plt.imshow(lum_img)
	imgplot.set_cmap('nipy_spectral')

	imgplot = plt.imshow(lum_img)
	plt.colorbar()

	if True:
		plt.tight_layout()
		plt.axis('off')
		plt.title('Image')
		plt.show()
	else:
		plt.imsave('./data/img_gray.png', lum_img, cmap='gray')

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
