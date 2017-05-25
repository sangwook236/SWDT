# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#histograms

import numpy as np

#%%-------------------------------------------------------------------
# PIL.

from PIL import Image

%img_array = np.uint8(img_array_float * 255)  # For RGB (3-dim)
%img_array = np.uint8(img_array_float * 255, mode='L')  # For grayscale (2-dim).

img = Image.fromarray(img_array)
img.show()
img.save("tmp.jpg")

#%%-------------------------------------------------------------------
# scipy.

import scipy.misc

scipy.misc.imshow(img_array)
scipy.misc.imsave('tmp.jpg', img_array)
