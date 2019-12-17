#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import skimage
import skimage.morphology
from skimage import io
import matplotlib.pyplot as plt

def plot_comparison(original, filtered, filter_name):
	fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
	ax1.imshow(original, cmap=plt.cm.gray)
	ax1.set_title('original')
	ax1.axis('off')
	ax2.imshow(filtered, cmap=plt.cm.gray)
	ax2.set_title(filter_name)
	ax2.axis('off')

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
def basic_example():
	orig_phantom = skimage.util.img_as_ubyte(skimage.data.shepp_logan_phantom())

	#fig, ax = plt.subplots()
	#ax.imshow(orig_phantom, cmap=plt.cm.gray)

	# Create a circular structuring element.
	#selem = skimage.morphology.ball(6)
	#selem = skimage.morphology.cube(12)
	#selem = skimage.morphology.diamond(6)
	selem = skimage.morphology.disk(6)
	#selem = skimage.morphology.square(6)
	#selem = skimage.morphology.octagon(6, 6)
	#selem = skimage.morphology.rectangle(6, 6)
	#selem = skimage.morphology.square(6)
	#selem = skimage.morphology.rectangle(3)

	# Erosion.
	# Morphological erosion sets a pixel at (i, j) to the minimum over all pixels in the neighborhood centered at (i, j).
	eroded = skimage.morphology.erosion(orig_phantom, selem)
	plot_comparison(orig_phantom, eroded, 'erosion')

	# Dilation.
	# Morphological dilation sets a pixel at (i, j) to the maximum over all pixels in the neighborhood centered at (i, j).
	# Dilation enlarges bright regions and shrinks dark regions.
	dilated = skimage.morphology.dilation(orig_phantom, selem)
	plot_comparison(orig_phantom, dilated, 'dilation')

	# Opening.
	# Morphological opening on an image is defined as an erosion followed by a dilation.
	# Opening can remove small bright spots (i.e. "salt") and connect small dark cracks.
	opened = skimage.morphology.opening(orig_phantom, selem)
	plot_comparison(orig_phantom, opened, 'opening')

	# Closing.
	# Morphological closing on an image is defined as a dilation followed by an erosion.
	# Closing can remove small dark spots (i.e. "pepper") and connect small bright cracks.
	phantom = orig_phantom.copy()
	phantom[10:30, 200:210] = 0

	closed = skimage.morphology.closing(phantom, selem)
	plot_comparison(phantom, closed, 'closing')

	# White tophat.
	# The white_tophat of an image is defined as the image minus its morphological opening.
	# This operation returns the bright spots of the image that are smaller than the structuring element.
	phantom = orig_phantom.copy()
	phantom[340:350, 200:210] = 255
	phantom[100:110, 200:210] = 0

	w_tophat = skimage.morphology.white_tophat(phantom, selem)
	plot_comparison(phantom, w_tophat, 'white tophat')

	# Black tophat.
	# The black_tophat of an image is defined as its morphological closing minus the original image.
	# This operation returns the dark spots of the image that are smaller than the structuring element.
	b_tophat = skimage.morphology.black_tophat(phantom, selem)
	plot_comparison(phantom, b_tophat, 'black tophat')

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
def simple_skeletonize_example():
	# Thinning is used to reduce each connected component in a binary image to a single-pixel wide skeleton.
	# It is important to note that this is performed on binary images only.

	horse = skimage.data.horse()

	#sk = skimage.morphology.skeletonize(horse == 0)  # Zhang's method.
	sk = skimage.morphology.skeletonize(horse == 0, method='lee')  # Lee's method.
	plot_comparison(horse, sk, 'skeletonize')

	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
def skeletonize_example():
	# Invert the horse image.
	image = skimage.util.invert(skimage.data.horse())

	# Perform skeletonization.
	skeleton = skimage.morphology.skeletonize(image)

	# Display results.
	fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),sharex=True, sharey=True)

	ax = axes.ravel()

	ax[0].imshow(image, cmap=plt.cm.gray)
	ax[0].axis('off')
	ax[0].set_title('original', fontsize=20)

	ax[1].imshow(skeleton, cmap=plt.cm.gray)
	ax[1].axis('off')
	ax[1].set_title('skeleton', fontsize=20)

	fig.tight_layout()
	plt.show()

	#--------------------
	data = skimage.data.binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

	skeleton = skimage.morphology.skeletonize(data)
	skeleton_lee = skimage.morphology.skeletonize(data, method='lee')

	fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(data, cmap=plt.cm.gray)
	ax[0].set_title('original')
	ax[0].axis('off')

	ax[1].imshow(skeleton, cmap=plt.cm.gray)
	ax[1].set_title('skeletonize')
	ax[1].axis('off')

	ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
	ax[2].set_title('skeletonize (Lee 94)')
	ax[2].axis('off')

	fig.tight_layout()
	plt.show()

	#--------------------
	# Medial axis skeletonization.

	# Generate the data
	data = skimage.data.binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

	# Compute the medial axis (skeleton) and the distance transform
	skel, distance = skimage.morphology.medial_axis(data, return_distance=True)

	# Compare with other skeletonization algorithms
	skeleton = skimage.morphology.skeletonize(data)
	skeleton_lee = skimage.morphology.skeletonize(data, method='lee')

	# Distance to the background for pixels of the skeleton
	dist_on_skel = distance * skel

	fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(data, cmap=plt.cm.gray)
	ax[0].set_title('original')
	ax[0].axis('off')

	ax[1].imshow(dist_on_skel, cmap='magma')
	ax[1].contour(data, [0.5], colors='w')
	ax[1].set_title('medial_axis')
	ax[1].axis('off')

	ax[2].imshow(skeleton, cmap=plt.cm.gray)
	ax[2].set_title('skeletonize')
	ax[2].axis('off')

	ax[3].imshow(skeleton_lee, cmap=plt.cm.gray)
	ax[3].set_title("skeletonize (Lee 94)")
	ax[3].axis('off')

	fig.tight_layout()
	plt.show()

	#--------------------
	# Morphological thinning.

	skeleton = skimage.morphology.skeletonize(image)
	thinned = skimage.morphology.thin(image)
	thinned_partial = skimage.morphology.thin(image, max_iter=25)

	fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(image, cmap=plt.cm.gray)
	ax[0].set_title('original')
	ax[0].axis('off')

	ax[1].imshow(skeleton, cmap=plt.cm.gray)
	ax[1].set_title('skeleton')
	ax[1].axis('off')

	ax[2].imshow(thinned, cmap=plt.cm.gray)
	ax[2].set_title('thinned')
	ax[2].axis('off')

	ax[3].imshow(thinned_partial, cmap=plt.cm.gray)
	ax[3].set_title('partially thinned')
	ax[3].axis('off')

	fig.tight_layout()
	plt.show()

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/applications/plot_morphology.html
def convex_hull_example():
	# The convex_hull_image is the set of pixels included in the smallest convex polygon that surround all white pixels in the input image.
	# Again note that this is also performed on binary images.

	horse = skimage.data.horse()

	hull1 = skimage.morphology.convex_hull_image(horse == 0)
	plot_comparison(horse, hull1, 'convex hull')

	horse_mask = horse == 0
	horse_mask[45:50, 75:80] = 1

	hull2 = skimage.morphology.convex_hull_image(horse_mask)
	plot_comparison(horse_mask, hull2, 'convex hull')

	plt.show()

def main():
	#basic_example()
	#simple_skeletonize_example()
	skeletonize_example()
	#convex_hull_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
