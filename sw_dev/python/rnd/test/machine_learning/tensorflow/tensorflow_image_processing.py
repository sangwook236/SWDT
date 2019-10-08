# REF [site] >> https://www.tensorflow.org/api_guides/python/image

import tensorflow as tf
from PIL import Image
import numpy as np

#--------------------------------------------------------------------

filenames = ['D:/dataset/pattern_recognition/street1.png', 'D:/dataset/pattern_recognition/street2.png']
#filenames = tf.train.match_filenames_once("D:/dataset/pattern_recognition/*.png")  # Not working.
#filenames = tf.train.match_filenames_once('D:/dataset/pattern_recognition/*.jpg')  # Not working.

count_num_files = tf.size(filenames)
filename_queue = tf.train.string_input_producer(filenames)

reader = tf.WholeFileReader()
_, img_value = reader.read(filename_queue)

#--------------------------------------------------------------------
# Encode and decode.

#rgb_img = tf.image.decode_png(img_value, channels = 3)
rgb_img = tf.image.decode_png(img_value)
#rgb_img = tf.image.decode_jpeg(img_value)

initializer = tf.global_variables_initializer()
with tf.Session() as sess:
	# Required to get the filename matching to run.
	sess.run(initializer)

	# Coordinate the loading of image files.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	# Get an image tensor and print its value.
	num_files = sess.run(count_num_files)
	for i in range(num_files):
		image = sess.run(rgb_img)  # numpy.ndarray.
		print(image.shape)
		#Image.fromarray(np.asarray(image)).save('temp.jpeg')
		Image.fromarray(np.asarray(image)).show()

	# Finish off the filename queue coordinator.
	coord.request_stop()
	coord.join(threads)

#tf.image.encode_png(rgb_img)
#tf.image.encode_jpeg(rgb_img)

#--------------------------------------------------------------------
# Converting between colorspaces.

rgb_img_float = tf.image.convert_image_dtype(rgb_img, tf.float32)
hsv_img = tf.image.rgb_to_hsv(tf.to_float(rgb_img))
