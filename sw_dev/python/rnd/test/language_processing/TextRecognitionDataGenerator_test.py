#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator

import os, random, glob, time
import numpy as np
import cv2
import trdg.generators, trdg.string_generator, trdg.utils

def visualize_generator(generator, output_mask, show_image):
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

def basic_example():
	lang = 'en'  # {'ar', 'cn', 'de', 'en', 'es', 'fr', 'hi'}.
	if True:
		num_examples = 10
	else:
		num_examples = -1
	if True:
		font_filepaths = trdg.utils.load_fonts(lang)
	else:
		font_filepaths = list()
	font_size = 32
	output_mask = True
	num_words = 5
	is_variable_length = True
	show_image = False

	generator_kwargs = {
		'language': lang,
		'count': num_examples,  # The number of examples.
		'fonts': font_filepaths, 'size': font_size,  # Font filepaths, font size.
		'skewing_angle': 0, 'random_skew': False,  # In degrees counter clockwise.
		#'blur': 0, 'random_blur': False,  # Blur radius.
		'blur': 2, 'random_blur': True,  # Blur radius.
		'distorsion_type': 0, 'distorsion_orientation': 0,  # distorsion_type = 0 (no distortion), 1 (sin), 2 (cos), 3 (random). distorsion_orientation = 0 (vertical), 1 (horizontal), 2 (both).
		'background_type': 0,  # background_type = 0 (Gaussian noise), 1 (plain white), 2 (quasicrystal), 3 (image).
		'width': -1,  # Specify a background width when width > 0.
		'alignment': 1,  # Align an image in a background image. alignment = 0 (left), 1 (center), the rest (right).
		'image_dir': None,  # Background image directory which is used when background_type = 3.
		'is_handwritten': False,
		#'text_color': '#282828',
		'text_color': '#000000,#FFFFFF',  # (0x00, 0x00, 0x00) ~ (0xFF, 0xFF, 0xFF).
		'orientation': 0,  # orientation = 0 (horizontal), 1 (vertical).
		'space_width': 1.0,  # The ratio of space width.
		'character_spacing': 0,  # Control space between characters (in pixels).
		'margins': (5, 5, 5, 5),  # For finer layout control. (top, left, bottom, right).
		'fit': False,  # For finer layout control. Specify if images and masks are cropped or not.
		'output_mask': output_mask,  # Specify if a character-level mask for each image is outputted or not.
		'word_split': False  # Split on word instead of per-character. This is useful for ligature-based languages.
	}

	# The generators use the same arguments as the CLI, only as parameters.
	if False:
		generator = trdg.generators.GeneratorFromDict(
			length=num_words,  # The number of words.
			allow_variable=is_variable_length,  # Is variable length?
			**generator_kwargs
		)
	elif False:
		use_letters, use_numbers, use_symbols = True, True, True

		generator = trdg.generators.GeneratorFromRandom(
			length=num_words,  # The number of words.
			allow_variable=is_variable_length,  # Is variable length?
			use_letters=use_letters, use_numbers=use_numbers, use_symbols=use_symbols,
			**generator_kwargs
		)
	elif False:
		min_num_words = 1

		generator = trdg.generators.GeneratorFromWikipedia(
			minimum_length=min_num_words,  # Min. number of words.
			**generator_kwargs
		)
	else:
		num_strings_to_generate = 1000

		if False:
			dictionary_filepath = './dictionary.txt'
			num_lines_to_read = 1000

			words = trdg.string_generator.create_strings_from_file(filename=dictionary_filepath, count=num_lines_to_read)

			#random.shuffle(words)
			strings = list()
			for _ in range(num_strings_to_generate):
				word_count = random.randint(1, num_words) if is_variable_length else num_words
				strings.append(' '.join(random.choices(words, k=word_count)))
		elif False:
			dictionary = trdg.utils.load_dict(lang)

			# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_dict.py
			strings = trdg.string_generator.create_strings_from_dict(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, lang_dict=dictionary)
		elif False:
			use_letters, use_numbers, use_symbols = True, True, True

			# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_random.py
			strings = trdg.string_generator.create_strings_randomly(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, let=use_letters, num=use_numbers, sym=use_symbols, lang=lang)
		elif False:
			min_num_words = 1

			# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_random.py
			strings = trdg.string_generator.create_strings_from_wikipedia(minimum_length=min_num_words, count=num_strings_to_generate, lang=lang)
		else:
			strings=['Test1', 'Test2', 'Test3']

		generator = trdg.generators.GeneratorFromStrings(
			strings=strings,
			**generator_kwargs
		)

	visualize_generator(generator, output_mask, show_image)

# REF [function] >> construct_charset() in ${SWL_PYTHON_HOME}/test/language_processing/text_generation_util.py.
def construct_charset(digit=True, alphabet_uppercase=True, alphabet_lowercase=True, punctuation=True, space=True, hangeul=True, hangeul_jamo=False, whitespace=False, unit=False, currency=False, latin=False, greek_uppercase=False, greek_lowercase=False, chinese=False, hiragana=False, katakana=False, hangeul_letter_filepath=None):
	charset = ''

	# Latin.
	# Unicode: Basic Latin (U+0020 ~ U+007F).
	import string
	if digit:
		charset += string.digits
	if alphabet_uppercase:
		charset += string.ascii_uppercase
	if alphabet_lowercase:
		charset += string.ascii_lowercase
	if punctuation:
		charset += string.punctuation
	if space:
		charset += ' '

	if hangeul:
		if 'posix' == os.name:
			work_dir_path = '/home/sangwook/work'
		else:
			work_dir_path = 'D:/work'
		swl_data_dir_path = work_dir_path + '/SWL_github/python/data'

		# Unicode: Hangul Syllables (U+AC00 ~ U+D7AF).
		#charset += ''.join([chr(ch) for ch in range(0xAC00, 0xD7A3 + 1)])

		if hangeul_letter_filepath is None:
			hangeul_letter_filepath = swl_data_dir_path + '/language_processing/hangul_ksx1001.txt'
			#hangeul_letter_filepath = swl_data_dir_path + '/language_processing/hangul_ksx1001_1.txt'
			#hangeul_letter_filepath = swl_data_dir_path + '/language_processing/hangul_unicode.txt'
		with open(hangeul_letter_filepath, 'r', encoding='UTF-8') as fd:
			#charset += fd.read().strip('\n')  # A string.
			charset += fd.read().replace(' ', '').replace('\n', '')  # A string.
			#charset += fd.readlines()  # A list of strings.
			#charset += fd.read().splitlines()  # A list of strings.
	if hangeul_jamo:
		# Unicode: Hangul Jamo (U+1100 ~ U+11FF), Hangul Compatibility Jamo (U+3130 ~ U+318F), Hangul Jamo Extended-A (U+A960 ~ U+A97F), & Hangul Jamo Extended-B (U+D7B0 ~ U+D7FF).
		##unicodes = list(range(0x1100, 0x11FF + 1)) + list(range(0x3131, 0x318E + 1))
		#unicodes = range(0x3131, 0x318E + 1)
		#charset += ''.join([chr(ch) for ch in unicodes])

		#charset += 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		charset += 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅛㅜㅠㅡㅣ'
		#charset += 'ㄱㄲㄳㄴㄵㄶㄷㄸㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅃㅄㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'

	if whitespace:
		#charset += '\n\t\v\b\r\f\a'
		charset += '\n\t\v\r\f'

	if unit:
		# REF [site] >> http://xahlee.info/comp/unicode_units.html
		unicodes = list(range(0x3371, 0x337A + 1)) + list(range(0x3380, 0x33DF + 1)) + [0x33FF]
		charset += ''.join([chr(ch) for ch in unicodes])

	if currency:
		# Unicode: Currency Symbols (U+20A0 ~ U+20CF).
		charset += ''.join([chr(ch) for ch in range(0x20A0, 0x20BF + 1)])

	if latin:
		# Unicode: Latin-1 Supplement (U+0080 ~ U+00FF), Latin Extended-A (U+0100 ~ U+017F), Latin Extended-B (U+0180 ~ U+024F).
		charset += ''.join([chr(ch) for ch in range(0x00C0, 0x024F + 1)])

	# Unicode: Greek and Coptic (U+0370 ~ U+03FF) & Greek Extended (U+1F00 ~ U+1FFF).
	if greek_uppercase:
		unicodes = list(range(0x0391, 0x03A1 + 1)) + list(range(0x03A3, 0x03A9 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])
	if greek_lowercase:
		unicodes = list(range(0x03B1, 0x03C1 + 1)) + list(range(0x03C3, 0x03C9 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])

	if chinese:
		# Unicode: CJK Unified Ideographs (U+4E00 ~ U+9FFF) & CJK Unified Ideographs Extension A (U+3400 ~ U+4DBF).
		unicodes = list(range(0x4E00, 0x9FD5 + 1)) + list(range(0x3400, 0x4DB5 + 1))
		charset += ''.join([chr(ch) for ch in unicodes])

	if hiragana:
		# Unicode: Hiragana (U+3040 ~ U+309F).
		charset += ''.join([chr(ch) for ch in range(0x3041, 0x3096 + 1)])
	if katakana:
		# Unicode: Katakana (U+30A0 ~ U+30FF).
		charset += ''.join([chr(ch) for ch in range(0x30A1, 0x30FA + 1)])

	return charset

# REF [function] >> construct_word_set() in ${SWL_PYTHON_HOME}/test/language_processing/text_generation_util.py.
def construct_word_set(korean=True, english=True, korean_dictionary_filepath=None, english_dictionary_filepath=None):
	if 'posix' == os.name:
		work_dir_path = '/home/sangwook/work'
	else:
		work_dir_path = 'D:/work'
	swl_data_dir_path = work_dir_path + '/SWL_github/python/data'

	words = []
	if korean:
		if korean_dictionary_filepath is None:
			korean_dictionary_filepath = swl_data_dir_path + '/language_processing/dictionary/korean_wordslistUnique.txt'

		print('Start loading a Korean dictionary...')
		start_time = time.time()
		with open(korean_dictionary_filepath, 'r', encoding='UTF-8') as fd:
			#korean_words = fd.readlines()
			#korean_words = fd.read().strip('\n')
			korean_words = fd.read().splitlines()
		print('End loading a Korean dictionary: {} secs.'.format(time.time() - start_time))
		words += korean_words
	if english:
		if english_dictionary_filepath is None:
			#english_dictionary_filepath = swl_data_dir_path + '/language_processing/dictionary/english_words.txt'
			english_dictionary_filepath = swl_data_dir_path + '/language_processing/wordlist_mono_clean.txt'
			#english_dictionary_filepath = swl_data_dir_path + '/language_processing/wordlist_bi_clean.txt'

		print('Start loading an English dictionary...')
		start_time = time.time()
		with open(english_dictionary_filepath, 'r', encoding='UTF-8') as fd:
			#english_words = fd.readlines()
			#english_words = fd.read().strip('\n')
			english_words = fd.read().splitlines()
		print('End loading an English dictionary: {} secs.'.format(time.time() - start_time))
		words += english_words

	return set(words)

# REF [function] >> generate_font_list() in ${SWL_PYTHON_HOME}/test/language_processing/text_generation_util.py.
def generate_font_list(font_filepaths):
	num_fonts = 1
	font_list = list()
	for fpath in font_filepaths:
		for font_idx in range(num_fonts):
			font_list.append((fpath, font_idx))

	return font_list

# REF [function] >> construct_font() in ${SWL_PYTHON_HOME}/test/language_processing/run_text_recognition.py.
def construct_font(korean=True, english=True):
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'

	font_dir_paths = list()
	if korean:
		font_dir_paths.append(font_base_dir_path + '/kor')
		#font_dir_paths.append(font_base_dir_path + '/kor_small')
		#font_dir_paths.append(font_base_dir_path + '/kor_large')
		#font_dir_paths.append(font_base_dir_path + '/kor_receipt')
	if english:
		font_dir_paths.append(font_base_dir_path + '/eng')
		#font_dir_paths.append(font_base_dir_path + '/eng_small')
		#font_dir_paths.append(font_base_dir_path + '/eng_large')
		#font_dir_paths.append(font_base_dir_path + '/eng_receipt')

	font_list = list()
	for dir_path in font_dir_paths:
		font_filepaths = glob.glob(os.path.join(dir_path, '*.ttf'))
		#font_list = generate_hangeul_font_list(font_filepaths)
		font_list.extend(generate_font_list(font_filepaths))
	return font_list

# REF [function] >> create_strings_randomly() in https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/string_generator.py.
def create_strings_randomly(length, allow_variable, count, let, num, sym, lang):
	"""
		Create all strings by randomly sampling from a pool of characters.
	"""

	# If none specified, use all three
	if True not in (let, num, sym):
		let, num, sym = True, True, True

	pool = ""
	if let:
		if lang == "kr":
			pool += construct_charset(digit=False, alphabet_uppercase=False, alphabet_lowercase=False, punctuation=False, space=False, hangeul=True)
			#pool += construct_charset(digit=False, alphabet_uppercase=True, alphabet_lowercase=True, punctuation=False, space=False, hangeul=True)
		elif lang == "cn":
			pool += "".join(
				[chr(i) for i in range(19968, 40908)]
			)  # Unicode range of CHK characters
		else:
			pool += string.ascii_letters
	if num:
		pool += "0123456789"
	if sym:
		pool += "!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"

	if lang == "kr":
		min_seq_len = 2
		max_seq_len = 10
	elif lang == "cn":
		min_seq_len = 1
		max_seq_len = 2
	else:
		min_seq_len = 2
		max_seq_len = 10

	strings = []
	for _ in range(0, count):
		current_string = ""
		for _ in range(0, random.randint(1, length) if allow_variable else length):
			seq_len = random.randint(min_seq_len, max_seq_len)
			current_string += "".join([random.choice(pool) for _ in range(seq_len)])
			current_string += " "
		strings.append(current_string[:-1])
	return strings

def korean_example():
	# NOTE [info] >> TextRecognitionDataGenerator does not support Korean.

	lang = 'kr'
	if True:
		num_examples = 10
	else:
		num_examples = -1
	if False:
		font_filepaths = trdg.utils.load_fonts(lang)
	else:
		font_filepaths = construct_font(korean=True, english=False)
		font_filepaths, _ = zip(*font_filepaths)
	font_size = 32
	output_mask = True
	num_words = 5
	is_variable_length = True
	num_strings_to_generate = 1000
	show_image = False

	if False:
		if False:
			# NOTE [warning] >> trdg.utils.load_dict() does not support Korean.
			dictionary = trdg.utils.load_dict(lang)
		else:
			dictionary = list(construct_word_set(korean=True, english=False))

		# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_dict.py
		strings = trdg.string_generator.create_strings_from_dict(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, lang_dict=dictionary)
	elif False:
		use_letters, use_numbers, use_symbols = True, True, True

		# REF [site] >> https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_random.py
		# NOTE [warning] >> trdg.string_generator.create_strings_randomly() does not support Korean.
		#	In order to support Korean in the function, we have to change it.
		#strings = trdg.string_generator.create_strings_randomly(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, let=use_letters, num=use_numbers, sym=use_symbols, lang=lang)
		strings = create_strings_randomly(length=num_words, allow_variable=is_variable_length, count=num_strings_to_generate, let=use_letters, num=use_numbers, sym=use_symbols, lang=lang)
	else:
		if 'posix' == os.name:
			work_dir_path = '/home/sangwook/work'
		else:
			work_dir_path = 'D:/work'
		swl_data_dir_path = work_dir_path + '/SWL_github/python/data'
		dictionary_filepath = swl_data_dir_path + '/language_processing/dictionary/korean_wordslistUnique.txt'
		num_lines_to_read = 366506

		words = trdg.string_generator.create_strings_from_file(filename=dictionary_filepath, count=num_lines_to_read)

		#random.shuffle(words)
		strings = list()
		for _ in range(num_strings_to_generate):
			word_count = random.randint(1, num_words) if is_variable_length else num_words
			strings.append(' '.join(random.choices(words, k=word_count)))

	generator = trdg.generators.GeneratorFromStrings(
		strings=strings,
		language=lang,
		count=num_examples,
		fonts=font_filepaths, size=font_size,
		skewing_angle=0, random_skew=False,
		blur=0, random_blur=False,
		distorsion_type=0, distorsion_orientation=0,
		background_type=0,
		width=-1,
		alignment=1,
		image_dir=None,
		is_handwritten=False,
		text_color='#000000,#FFFFFF',
		orientation=0,
		space_width=1.0,
		character_spacing=0,
		margins=(5, 5, 5, 5),
		fit=False,
		output_mask=output_mask,
		word_split=False
	)

	visualize_generator(generator, output_mask, show_image)

def main():
	#basic_example()

	korean_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
