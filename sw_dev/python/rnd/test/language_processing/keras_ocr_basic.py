#!/usr/bin/env python
# coding: UTF-8

import os, string
import numpy as np
import keras_ocr
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/faustomorales/keras-ocr
def detection_and_recognition_test():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'

	if True:
		data_dir_path = data_base_dir_path + '/text/receipt_icdar2019'

		image_filepaths = [
			data_dir_path + '/0325updated.task1train(626p)-20190531T071023Z-001/0325updated.task1train(626p)/X00016469612.jpg',
		]
	elif False:
		data_dir_path = data_base_dir_path + '/text/receipt_epapyrus/epapyrus_20190618'

		image_filepaths = list()
		for idx in range(1, 11):
			image_filepaths.append(data_dir_path + '/receipt_1/img{:02}.jpg'.format(idx))
		for idx in range(1, 32):
			image_filepaths.append(data_dir_path + '/receipt_2/img{:02}.jpg'.format(idx))
		for idx in range(1, 40):
			image_filepaths.append(data_dir_path + '/receipt_3/img{:02}.jpg'.format(idx))
	elif False:
		data_dir_path = data_base_dir_path + '/text/receipt_epapyrus/keit_20190619'

		image_filepaths = list()
		for idx in range(1, 6):
			image_filepaths.append(data_dir_path + '카드영수증_{}-1.png'.format(idx))
		image_filepaths.extend([
			data_dir_path + '/tax_invoice_01/tax_invoice_01.jpg',
			data_dir_path + '/tax_invoice_01/tax_invoice_02.jpg',
			data_dir_path + '/tax_invoice_01/tax_invoice_03.jpg',
			data_dir_path + '/tax_invoice_01/tax_invoice_04.jpg',
			data_dir_path + '/tax_invoice_02/tax_invoice_01.jpg',
			data_dir_path + '/tax_invoice_02/tax_invoice_02.jpg',
			data_dir_path + '/tax_invoice_02/tax_invoice_03.jpg',
			data_dir_path + '/tax_invoice_03/tax_invoice_01.jpg',
			data_dir_path + '/tax_invoice_04/tax_invoice_01.jpg',
			data_dir_path + '/tax_invoice_05/tax_invoice_01.jpg',
		])
		for idx in range(1, 7):
			image_filepaths.append(data_dir_path + 'import_license_01/import_license_{:02}.png'.format(idx))

	if True:
		detector = keras_ocr.detection.Detector(pretrained=True)
	else:
		detector = keras_ocr.detection.Detector(pretrained=False)
		detector.model.load_weights('./craft_mlt_25k.h5')
		# REF [function] >> training_test().
		#detector.model.load_weights('./v0_detector.h5')  # FIXME [fix] >> Not correctly working.

	"""
	# The alphabet defines which characters the OCR will be trained to detect.
	alphabet = string.digits + \
			   string.ascii_lowercase + \
			   string.ascii_uppercase + \
			   string.punctuation + ' '

	recognizer = keras_ocr.recognition.Recognizer(
		alphabet=alphabet,
		width=128,
		height=64
	)
	# REF [function] >> training_test().
	recognizer.model.load_weights('./v0_recognizer.h5')  # FIXME [fix] >> Not correctly working.
	"""

	for image_filepath in image_filepaths:
		image = keras_ocr.tools.read(image_filepath)

		# Boxes will be an Nx4x2 array of box quadrangles, where N is the number of detected text boxes.
		boxes = detector.detect(images=[image])[0]
		canvas = keras_ocr.detection.drawBoxes(image, boxes)
		plt.imshow(canvas)
		plt.show()

		"""
		boxes = detector.detect(images=[image])[0]
		predictions = recognizer.recognize_from_boxes(boxes=boxes, image=image)
		print('Predictions = ', predictions)
		"""

# REF [site] >> https://github.com/faustomorales/keras-ocr
def training_test():
	# The alphabet defines which characters the OCR will be trained to detect.
	alphabet = string.digits + \
			   string.ascii_lowercase + \
			   string.ascii_uppercase + \
			   string.punctuation + ' '

	# Build the text detector (pretrained).
	detector = keras_ocr.detection.Detector(pretrained=True)
	detector.model.compile(
		loss='mse',
		optimizer='adam'
	)

	# Build the recognizer (randomly initialized) and build the training model.
	recognizer = keras_ocr.recognition.Recognizer(
		alphabet=alphabet,
		width=128,
		height=64
	)
	recognizer.create_training_model(max_string_length=16)

	# For each text sample, the text generator provides
	# a list of (category, string) tuples. The category
	# is used to select which fonts the image generator
	# should choose from when rendering those characters 
	# (see the image generator step below) this is useful
	# for cases where you have characters that are only
	# available in some fonts. You can replace this with
	# your own generator, just be sure to match
	# that function signature if you are using
	# recognizer.get_image_generator. Alternatively,
	# you can provide your own image_generator altogether.
	# The default text generator uses the DocumentGenerator
	# from essential-generators.
	detection_text_generator = keras_ocr.tools.get_text_generator(
		max_string_length=32,
		alphabet=alphabet
	)

	# The image generator generates (image, sentence, lines)
	# tuples where image is a HxWx3 image, 
	# sentence is a string using only letters
	# from the selected alphabet, and lines is a list of
	# lines of text in the image where each line is a list of 
	# tuples of the form (x1, y1, x2, y2, x3, y3, y4, c). c
	# is the character in the line and (x1, y2), (x2, y2), (x3, y3),
	# (x4, y4) define the bounding coordinates in clockwise order
	# starting from the top left. You can replace
	# this with your own generator, just be sure to match
	# that function signature.
	detection_image_generator = keras_ocr.tools.get_image_generator(
		height=256,
		width=256,
		x_start=(10, 30),
		y_start=(10, 30),
		single_line=False,
		text_generator=detection_text_generator,
		font_groups={
			'characters': [
				'Century Schoolbook',
				'Courier',
				'STIX',
				'URW Chancery L',
				'FreeMono'
			]
		}
	)

	# From our image generator, create a training batch generator and train the model.
	detection_batch_generator = detector.get_batch_generator(
		image_generator=detection_image_generator,
		batch_size=2,
	)
	detector.model.fit_generator(
	  generator=detection_batch_generator,
	  steps_per_epoch=100,
	  epochs=10,
	  workers=0
	)
	detector.model.save_weights('v0_detector.h5')

	# This next part is similar to before but now
	# we adjust the image generator to provide only
	# single lines of text.
	recognition_image_generator = keras_ocr.tools.convert_multiline_generator_to_single_line(
		multiline_generator=detection_image_generator,
		max_string_length=recognizer.training_model.input_shape[1][1],
		target_width=recognizer.model.input_shape[2],
		target_height=recognizer.model.input_shape[1]
	)
	recognition_batch_generator = recognizer.get_batch_generator(
		image_generator=recognition_image_generator,
		batch_size=8
	)
	recognizer.training_model.fit_generator(
		generator=recognition_batch_generator,
		steps_per_epoch=100,
		epochs=100
	)

	# You can save the model weights to use later.
	recognizer.model.save_weights('v0_recognizer.h5')

	# Once training is done, you can use recognize to extract text.
	#image, _, _ = next(detection_image_generator)
	if 'posix' == os.name:
		receipt_icdar2019_base_dir_path = '/home/sangwook/work/dataset/text/receipt_icdar2019'
	else:
		receipt_icdar2019_base_dir_path = 'D:/work/dataset/text/receipt_icdar2019'
	#image_filepath = receipt_icdar2019_base_dir_path + '/image/00000.jpg'
	image_filepath = receipt_icdar2019_base_dir_path + '/0325updated.task1train(626p)-20190531T071023Z-001/0325updated.task1train(626p)/X00016469612.jpg'

	image = keras_ocr.tools.read(image_filepath)

	boxes = detector.detect(images=[image])[0]
	predictions = recognizer.recognize_from_boxes(boxes=boxes, image=image)
	print('Predictions =', predictions)

def main():
	detection_and_recognition_test()

	#training_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
