#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import skimage
import skimage.measure
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.widgets.history import SaveButtons
from skimage.viewer.plugins.lineprofile import LineProfile
from skimage.viewer.plugins.base import Plugin
import pandas as pd

# REF [site] >> http://scikit-image.org/docs/stable/user_guide/viewer.html
def viewer():
	image = skimage.data.coins()

	viewer = ImageViewer(image)
	viewer.show()

	viewer += LineProfile(viewer)
	overlay, data = viewer.show()[0]

	#--------------------
	denoise_plugin = Plugin(image_filter=skimage.restoration.denoise_tv_bregman)

	denoise_plugin += Slider('weight', 0.01, 0.5, update_on='release')
	denoise_plugin += SaveButtons()

	viewer = ImageViewer(image)
	viewer += denoise_plugin
	denoised = viewer.show()[0][0]

# REF [site] >> https://scikit-image.org/docs/dev/api/skimage.measure.html
def regionprops_example():
	img = skimage.util.img_as_ubyte(skimage.data.coins()) > 110
	label_img = skimage.measure.label(img, connectivity=img.ndim)

	props = skimage.measure.regionprops(label_img, intensity_image=None, cache=True)

	print('#labeled objects =', len(props))

	# Properties:
	# 	area, bbox, bbox_area, centroid, convex_area, convex_image, coords, eccentricity, equivalent_diameter, euler_number,
	#	extent, filled_area, filled_image, image, inertia_tensor, inertia_tensor_eigvals, intensity_image, label
	#	local_centroid, major_axis_length, max_intensity, mean_intensity, min_intensity, minor_axis_length, moments,
	#	moments_central, moments_hu, moments_normalized, orientation, perimeter, slice, solidity, weighted_centroid,
	#	weighted_local_centroid, weighted_moments, weighted_moments_central, weighted_moments_hu, weighted_moments_normalized.

	# Label of first labeled object.
	print('Label =', props[0].label)
	#print('Label =', props[0]['label'])

	# Centroid of first labeled object.
	print('Centroid =', props[0].centroid)

# REF [site] >> https://scikit-image.org/docs/dev/api/skimage.measure.html
def regionprops_table_example():
	image = skimage.data.coins()
	label_image = skimage.measure.label(image > 110, connectivity=image.ndim)

	props = skimage.measure.regionprops_table(label_image, image, properties=['label', 'inertia_tensor', 'inertia_tensor_eigvals'], cache=True, separator='-')

	data = pd.DataFrame(props)
	print('Property =', data.head())

def main():
	#viewer()

	# Image statistics/properties.
	regionprops_example()
	#regionprops_table_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
