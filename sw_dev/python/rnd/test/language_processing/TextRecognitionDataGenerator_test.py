#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator

import numpy as np
import cv2
import trdg.generators

def basic_example():
	output_mask = True
	show_image = False

	generator_kwargs = {
		#'count': -1,  # The number of examples.
		'count': 10,  # The number of examples.
		'language': 'en',
		'fonts': [], 'size': 32,  # Font filepaths, font size.
		'skewing_angle': 0, 'random_skew': False,
		#'blur': 0, 'random_blur': False,
		'blur': 2, 'random_blur': True,
		'distorsion_type': 0, 'distorsion_orientation': 0,
		'background_type': 0,
		'width': -1,  # Specify a background width when width > 0.
		'alignment': 1,
		'image_dir': None,  # Background image directory which is used when background_type = 3.
		'is_handwritten': False,
		#'text_color': '#282828',
		'text_color': '#000000,#FFFFFF',  # (0x00, 0x00, 0x00) ~ (0xFF, 0xFF, 0xFF).
		'orientation': 0,  # Specify if text orientation is horizontal (orientation = 0) or vertical (orientation = 1).
		'space_width': 1.0,  # The ratio of space width.
		'character_spacing': 0,  # Control space between characters (in pixels).
		'margins': (5, 5, 5, 5),  # For finer layout control.
		'fit': False,  # For finer layout control. Specify if images and masks are cropped or not.
		'output_mask': output_mask,  # Specify if a character-level mask for each image is outputted or not.
		'word_split': False  # Split on word instead of per-character. This is useful for ligature-based languages.
	}

	# The generators use the same arguments as the CLI, only as parameters.
	if False:
		generator = trdg.generators.GeneratorFromDict(
			length=5,  # The number of words.
			allow_variable=True,  # Is variable length?
			**generator_kwargs
		)
	elif False:
		generator = trdg.generators.GeneratorFromRandom(
			length=3,  # The number of words.
			allow_variable=False,  # Is variable length?
			use_letters=True, use_numbers=True, use_symbols=True,
			**generator_kwargs
		)
	elif False:
		generator = trdg.generators.GeneratorFromWikipedia(
			minimum_length=1,  # Min. number of words.
			**generator_kwargs
		)
	else:
		generator = trdg.generators.GeneratorFromStrings(
			strings=['Test1', 'Test2', 'Test3'],
			**generator_kwargs
		)

	if output_mask:
		for idx, ((img, msk), lbl) in enumerate(generator):
			print('Image: type = {}, size = {}, mode = {}.'.format(type(img), img.size, img.mode))
			print('Mask:  type = {}, size = {}, mode = {}.'.format(type(msk), msk.size, msk.mode))
			print('Label: type = {}, len = {}.'.format(type(lbl), len(lbl)))

			print('Label = {}.'.format(lbl))
			if show_image:
				#img.show()
				img = np.asarray(img, dtype=np.uint8)
				msk = np.asarray(msk, dtype=np.int16)
				print('Image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(img.shape, img.dtype, np.min(img), np.max(img)))
				print('Mask:  shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(msk.shape, msk.dtype, np.min(msk), np.max(msk)))
				#cv2.imshow('image_{}.png'.format(idx), img)
				#cv2.imshow('mask_{}.png'.format(idx), img)
				cv2.imshow('Input', img)
				cv2.imshow('Mask', msk)
				cv2.waitKey(0)
	else:	
		for idx, (img, lbl) in enumerate(generator):
			print('Image: type = {}, size = {}, mode = {}.'.format(type(img), img.size, img.mode))
			print('Label: type = {}, len = {}.'.format(type(lbl), len(lbl)))

			print('Label = {}.'.format(lbl))
			if show_image:
				#img.show()
				img = np.asarray(img, dtype=np.uint8)
				#cv2.imshow('image_{}.png'.format(idx), img)
				cv2.imshow('Input', img)
				cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
