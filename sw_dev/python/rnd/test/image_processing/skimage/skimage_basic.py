#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import Slider
from skimage.viewer.widgets.history import SaveButtons
from skimage.viewer.plugins.lineprofile import LineProfile
from skimage.viewer.plugins.base import Plugin
from skimage.restoration import denoise_tv_bregman

# REF [site] >> http://scikit-image.org/docs/stable/user_guide/viewer.html
def viewer():
	image = data.coins()

	viewer = ImageViewer(image)
	viewer.show()

	viewer += LineProfile(viewer)
	overlay, data = viewer.show()[0]

	#--------------------
	denoise_plugin = Plugin(image_filter=denoise_tv_bregman)

	denoise_plugin += Slider('weight', 0.01, 0.5, update_on='release')
	denoise_plugin += SaveButtons()

	viewer = ImageViewer(image)
	viewer += denoise_plugin
	denoised = viewer.show()[0][0]

def main():
	viewer()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
