# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#histograms

import numpy as np

#%%-------------------------------------------------------------------
# PIL.

from PIL import Image

img = Image.open('/path/to/image')

img_array = np.asarray(img, dtype='uint8')
#img_array = np.uint8(img_array_float * 255)  # For RGB (3-dim).
#img_array = np.uint8(img_array_float * 255, mode='L')  # For grayscale (2-dim).

# Do something.

img = Image.fromarray(img_array)
img.show()
img.save("tmp.jpg")

#%%-------------------------------------------------------------------
# scipy.

import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt

img_array = scipy.ndimage.imread('/path/to/image', mode="RGB")

#scipy.misc.imshow(img_array)  # Use the environment variable, SCIPY_PIL_IMAGE_VIEWER.
plt.imshow(img_array)
scipy.misc.imsave('tmp.jpg', img_array)
