#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator

import time
import trdg.generators

def basic_example():
	# The generators use the same arguments as the CLI, only as parameters
	generator = trdg.generators.GeneratorFromStrings(
		strings=['Test1', 'Test2', 'Test3'],
		#count=-1,  # The number of examples
		count=10,  # The number of examples
		language='en', fonts=[], size=32,  # Language, fonts, font size.
		skewing_angle=0, random_skew=False,
		#blur=0, random_blur=False,
		blur=2, random_blur=True,
		background_type=0,
		distorsion_type=0, distorsion_orientation=0,
		is_handwritten=False,
		width=-1,
		alignment=1,
		text_color='#282828',
		orientation=0,
		space_width=1.0,
		character_spacing=0,
		margins=(5, 5, 5, 5),
		fit=False,
		output_mask=False,
		word_split=False,
		image_dir=None  # Background image directory which is used when background_type = 3.
	)
	"""
	generator = trdg.generators.GeneratorFromDict(
		length=1,  # The number of words.
		allow_variable=False,  # Is variable length?
		...
	)
	generator = trdg.generators.GeneratorFromRandom(
		length=1,  # The number of words.
		allow_variable=False,  # Is variable length?
		use_letters=True, use_numbers=True, use_symbols=True,
		...
	)
	generator = trdg.generators.GeneratorFromWikipedia(
		minimum_length=1,  # Min. number of words.
		...
	)
	"""

	for img, lbl in generator:
		print('Image: type = {}, size = {}, mode = {}.'.format(type(img), img.size, img.mode))
		print('Label: type = {}, len = {}.'.format(type(lbl), len(lbl)))

		print('Label = {}.'.format(lbl))
		img.show()

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
