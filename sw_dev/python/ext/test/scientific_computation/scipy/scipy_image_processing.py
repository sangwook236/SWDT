# REF [site] >> https://docs.scipy.org/doc/scipy-0.18.1/reference/misc.html

import scipy.ndimage, scipy.misc
import numpy as np

#%%-------------------------------------------------------------------
# Morphological operation.

img_filename = 'D:/dataset/phenotyping/RDA/all_plants_foreground/adaptor1_side_120_foreground.png'

img = scipy.misc.imread(img_filename, mode='L')

img_eroded = scipy.ndimage.grey_erosion(img, size=(3, 3))

#footprint = scipy.ndimage.generate_binary_structure(2, 2)
#img_eroded = scipy.ndimage.grey_erosion(img, size=(3, 3), footprint=footprint)

#scipy.misc.imshow(img_eroded)
scipy.misc.imsave('tmp.jpg', img_eroded)
