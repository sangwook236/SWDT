# REF [site] >> https://docs.scipy.org/doc/numpy/user/quickstart.html#histograms

import numpy as np

#%%-------------------------------------------------------------------
# PIL.

from PIL import Image

img = Image.open('/path/to/image')

img_array = np.asarray(img, dtype=np.uint8)
#img_array = (img_array_float * 255).astype(np.uint8)

# Do something.

img = Image.fromarray(img_array)
#img_gray = Image.fromarray(img_array, mode='L')
#img_rgb = Image.fromarray(img_array, mode='RGB')
img.show()
img.save('./tmp.jpg')

img_array_float = np.asarray(img, dtype=np.float)
img_ubyte = Image.fromarray((img_array_float * 255).astype(np.uint8), mode='L')
img_ubyte.save('./tmp_ubyte.png')
img_float = Image.fromarray(img_array_float.astype(np.float32), mode='F')
img_float.save('./tmp_float.tif')

#%%-------------------------------------------------------------------
# scipy.

import scipy.ndimage
import scipy.misc

img_array = scipy.ndimage.imread('/path/to/image', mode='RGB')
#scipy.misc.imshow(img_array)  # Use the environment variable, SCIPY_PIL_IMAGE_VIEWER.
scipy.misc.imsave('./tmp.jpg', img_array)

#%%-------------------------------------------------------------------
# matplotlib.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img_array = mpimg.imread('/path/to/image')
plt.imshow(img_array)
plt.imsave('./tmp.png', img_array)
#plt.imsave('./tmp_gray.png', img_array, cmap='gray')
