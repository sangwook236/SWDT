# REF [site] >> https://www.tensorflow.org/api_guides/python/image

import tensorflow as tf
from PIL import Image
import numpy as np

#%%------------------------------------------------------------------

#filenames = ['D:/dataset/pattern_recognition/street1.png']
filenames = tf.train.match_filenames_once('D:/dataset/pattern_recognition/*.png')
#filenames = tf.train.match_filenames_once('D:/dataset/pattern_recognition/*.jpg')

count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
key, img_value = reader.read(filename_queue)

#%%------------------------------------------------------------------
# Encode and decode.

#rgb_img = tf.image.decode_png(img_value, channels = 3)
rgb_img = tf.image.decode_png(img_value)
#rgb_img = tf.image.decode_jpeg(img_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	num_files = sess.run(count_num_files)
	for i in range(num_files):
		image = rgb_img.eval()
		print(image.shape)
		#Image.fromarray(np.asarray(image)).save('temp.jpeg')
		Image.fromarray(np.asarray(image)).show()

tf.image.encode_png(rgb_img)
tf.image.encode_jpeg(rgb_img)

#%%------------------------------------------------------------------
# Converting between colorspaces.

rgb_img_float = tf.image.convert_image_dtype(rgb_img, tf.float32)
hsv_img = tf.image.rgb_to_hsv(rgb_img)
