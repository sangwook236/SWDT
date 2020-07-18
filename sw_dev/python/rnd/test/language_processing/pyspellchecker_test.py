#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/barrust/pyspellchecker
#	http://norvig.com/spell-correct.html

import time
import spellchecker

# REF [site] >> https://github.com/barrust/pyspellchecker
def simple_example():
	spell = spellchecker.SpellChecker(language='en',  # Supported languages: 'en', 'es', 'de', 'fr' and 'pt'. Defaults to 'en'.
		local_dictionary=None,  # The path to a locally stored word frequency dictionary. If provided, no language will be loaded.
		distance=2,  # The edit distance to use. Defaults to 2.
		tokenizer=None,
		case_sensitive=False)

	# Find those words that may be misspelled.
	misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

	for word in misspelled:
		# Get the one 'most likely' answer.
		print('spell.correction({}) = {}.'.format(word, spell.correction(word)))

		# Get a list of 'likely' options.
		print('spell.candidates({}) = {}.'.format(word, spell.candidates(word)))

	print("spell.word_probability('here') = {}.".format(spell.word_probability('here')))

	#--------------------
	# If the Word Frequency list is not to your liking, you can add additional text to generate a more appropriate list for your use case.
	spell = spellchecker.SpellChecker()  # Loads default word frequency list.

	"""
	# In my_dictionary.json
	#	{
	#		"a": 1,
	#		"b": 2,
	#		"apple": 45,
	#		"bike": 60
	#	}
	dictionary_filepath = './my_dictionary.json'
	spell.word_frequency.load_dictionary(dictionary_filepath, encoding='UTF-8')
	"""
	"""
	text_filepath = './my_text.txt'
	spell.word_frequency.load_text_file(text_filepath, encoding='UTF-8')
	"""
	text_data = "A blue whale went for a swim in the sea. Along it's path it ran into a storm. To avoid the storm it dove deep under the waves."
	spell.word_frequency.load_text(text_data)

	# If I just want to make sure some words are not flagged as misspelled.
	spell.word_frequency.load_words(['microsoft', 'apple', 'google'])
	print("spell.known(['microsoft', 'google', 'facebook']) = {}.".format(spell.known(['microsoft', 'google', 'facebook'])))  # Will return both now!

	print('len(spell.word_frequency.dictionary) = {}.'.format(len(spell.word_frequency.dictionary)))
	print('spell.word_frequency.total_words = {}.'.format(spell.word_frequency.total_words))
	print('spell.word_frequency.unique_words = {}.'.format(spell.word_frequency.unique_words))
	print('len(spell.word_frequency.letters) = {}.'.format(len(spell.word_frequency.letters)))
	print('spell.word_frequency.longest_word_length = {}.'.format(spell.word_frequency.longest_word_length))

	print('spell.word_frequency.tokenize(text_data)) = {}.'.format(list(spell.word_frequency.tokenize(text_data))))

	print('spell.word_frequency.keys()) = {}.'.format(list(word for idx, word in enumerate(spell.word_frequency.keys()) if idx < 20)))
	print('spell.word_frequency.words()) = {}.'.format(list(word for idx, word in enumerate(spell.word_frequency.words()) if idx < 20)))
	print('spell.word_frequency.items()) = {}.'.format(list(word for idx, word in enumerate(spell.word_frequency.items()) if idx < 20)))

	#--------------------
	# If the words that you wish to check are long, it is recommended to reduce the distance to 1.

	spell = spellchecker.SpellChecker(distance=1)  # Set at initialization.

	# Do some work on longer words.

	spell.distance = 2  # Set the distance parameter back to the default.

def construct_korean_dictionary_example():
	import konlpy
	#import nltk

	text_filepath = './korean_modern_novel_1_2.txt'
	dictionary_filepath = './my_korean_dictionary.json'

	print('Start loading a text file...')
	start_time = time.time()
	try:
		with open(text_filepath, 'r', encoding='UTF-8') as fd:
			text_data = fd.read()
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(text_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(text_filepath))
		return
	print('End loading a text file: {} secs.'.format(time.time() - start_time))

	print('Start constructing a Korean dictionary...')
	start_time = time.time()
	#kkma = konlpy.tag.Kkma()
	#text_data = kkma.nouns(text_data)
	okt = konlpy.tag.Okt()
	text_data = okt.nouns(text_data)
	print('End constructing a Korean dictionary: {} secs.'.format(time.time() - start_time))

	# In my_korean_dictionary.json:
	#	{
	#		"가": 1,
	#		"나": 2,
	#		"사과": 45,
	#		"자전거": 60
	#	}

	print('Start saving a Korean dictionary...')
	start_time = time.time()
	spell = spellchecker.SpellChecker(language=None)
	text_data = ' '.join(text_data)
	spell.word_frequency.load_text(text_data)
	spell.export(dictionary_filepath, encoding='UTF-8', gzipped=True)
	print('End saving a Korean dictionary: {} secs.'.format(time.time() - start_time))

	print('len(spell.word_frequency.dictionary) = {}.'.format(len(spell.word_frequency.dictionary)))
	print('spell.word_frequency.total_words = {}.'.format(spell.word_frequency.total_words))
	print('spell.word_frequency.unique_words = {}.'.format(spell.word_frequency.unique_words))
	print('len(spell.word_frequency.letters) = {}.'.format(len(spell.word_frequency.letters)))
	print('spell.word_frequency.longest_word_length = {}.'.format(spell.word_frequency.longest_word_length))

def simple_korean_example():
	spell = spellchecker.SpellChecker(language='en', distance=2)  # Loads default word frequency list.

	dictionary_filepath = './my_korean_dictionary.json'
	spell.word_frequency.load_dictionary(dictionary_filepath, encoding='UTF-8')
	"""
	text_filepath = './korean_modern_novel_1_2.txt'
	spell.word_frequency.load_text_file(text_filepath, encoding='UTF-8')
	"""

	# Find those words that may be misspelled.
	misspelled = spell.unknown(['천재즈변', '학교', '도시관', '도소관', '요기'])

	for word in misspelled:
		# Get the one 'most likely' answer.
		print('spell.correction({}) = {}.'.format(word, spell.correction(word)))

		# Get a list of 'likely' options.
		print('spell.candidates({}) = {}.'.format(word, spell.candidates(word)))

	print("spell.word_probability('학교') = {}.".format(spell.word_probability('학교')))

	#text_data = "A blue whale went for a swim in the sea. Along it's path it ran into a storm. To avoid the storm it dove deep under the waves."
	#spell.word_frequency.load_text(text_data)

	# If I just want to make sure some words are not flagged as misspelled.
	spell.word_frequency.load_words(['마이크로소프트', '애플', '구글'])
	print("spell.known(['마이크로소프트', '구글', '페이스북']) = {}.".format(spell.known(['마이크로소프트', '구글', '페이스북'])))  # Will return both now!
	spell.word_frequency.load_words(['microsoft', 'apple', 'google'])
	print("spell.known(['microsoft', 'google', 'facebook']) = {}.".format(spell.known(['microsoft', 'google', 'facebook'])))  # Will return both now!

	print('len(spell.word_frequency.dictionary) = {}.'.format(len(spell.word_frequency.dictionary)))
	print('spell.word_frequency.total_words = {}.'.format(spell.word_frequency.total_words))
	print('spell.word_frequency.unique_words = {}.'.format(spell.word_frequency.unique_words))
	print('len(spell.word_frequency.letters) = {}.'.format(len(spell.word_frequency.letters)))
	print('spell.word_frequency.longest_word_length = {}.'.format(spell.word_frequency.longest_word_length))

def correct_text():
	korean_dictionary_filepath = './my_korean_dictionary.json'
	text_filepath = './Sample-2_0.txt'

	#--------------------
	print('Start loading dictionary...')
	start_time = time.time()
	spell = spellchecker.SpellChecker(language='en', distance=1)  # Loads default word frequency list.

	spell.word_frequency.load_dictionary(korean_dictionary_filepath, encoding='UTF-8')
	print('End loading dictionary: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start loading a text file...')
	start_time = time.time()
	try:
		with open(text_filepath, 'r', encoding='UTF-8') as fd:
			data = fd.read()
			words = data.split()
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(text_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(text_filepath))
		return
	print('End loading a text file: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Find those words that may be misspelled.
	misspelled = spell.unknown(words)

	print('Words = {}.'.format(words))
	print('Misspelled words = {}.'.format(misspelled))

	for word in misspelled:
		print("Start correcting a word, '{}'...".format(word))
		start_time = time.time()

		# Get the one 'most likely' answer.
		print('spell.correction({}) = {}.'.format(word, spell.correction(word)))

		# Get a list of 'likely' options.
		print('spell.candidates({}) = {}.'.format(word, spell.candidates(word)))

		print('End correcting a word: {} secs.'.format(time.time() - start_time))

def correct_text2():
	korean_dictionary_filepath = './my_korean_dictionary.json'
	text_filepath = './Sample-0_0.txt'

	#--------------------
	print('Start loading dictionary...')
	start_time = time.time()
	spell = spellchecker.SpellChecker(language='en', distance=1)  # Loads default word frequency list.

	spell.word_frequency.load_dictionary(korean_dictionary_filepath, encoding='UTF-8')

	#spell.word_frequency.load_words(['grit', '고형분', '체상량', '미립겔', '자성이물'])
	print('End loading dictionary: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start loading a text file...')
	start_time = time.time()
	try:
		with open(text_filepath, 'r', encoding='UTF-8') as fd:
			data = fd.read()
			words = data.split()
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(text_filepath))
		return
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(text_filepath))
		return
	print('End loading a text file: {} secs.'.format(time.time() - start_time))

	#--------------------
	for word in words:
		# Find those words that may be misspelled.
		misspelled = spell.unknown([word])
		if misspelled:
			print("Start correcting a word, '{}'...".format(word))
			start_time = time.time()
			for msw in misspelled:
				# Get the one 'most likely' answer.
				print('spell.correction({}) = {}.'.format(msw, spell.correction(msw)))

				# Get a list of 'likely' options.
				#print('spell.candidates({}) = {}.'.format(msw, spell.candidates(msw)))
			print('End correcting a word: {} secs.'.format(time.time() - start_time))

def main():
	#simple_example()

	#construct_korean_dictionary_example()
	#simple_korean_example()

	#correct_text()
	correct_text2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
