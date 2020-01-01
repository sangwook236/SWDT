#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import collections, itertools, re
import symspellpy
import sklearn.datasets
import pkg_resources

# REF [site] >> https://symspellpy.readthedocs.io/en/latest/examples/dictionary.html
def dictionary_example():
	#--------------------
	# Load frequency dictionary.

	# Given a dictionary file like:
	#	<term> <count>
	#	<term> <count>
	#	...
	#	<term> <count>

	dictionary_filepath = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
	symspell = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7, count_threshold=1)
	symspell.load_dictionary(dictionary_filepath, term_index=0, count_index=1)

	# Print out first 5 elements to demonstrate that dictionary is successfully loaded.
	print(list(itertools.islice(symspell.words.items(), 5)))

	# Given a bigram dictionary file like:
	#	<term_part_1> <term_part_2> <count>
	#	<term_part_1> <term_part_2> <count>
	#	...
	#	<term_part_1> <term_part_2> <count>

	dictionary_filepath = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')
	symspell = symspellpy.SymSpell()
	symspell.load_bigram_dictionary(dictionary_filepath, 0, 2)

	# Print out first 5 elements to demonstrate that dictionary is successfully loaded.
	print(list(itertools.islice(symspell.bigrams.items(), 5)))

	#--------------------
	# Load frequency dictionary with custom separator.

	dictionary_filepath = './dictionary.txt'
	dictionary_data = """the$23135851162
abcs of$10956800
of$13151942776
aaron and$10721728
abbott and$7861376
abbreviations and$13518272
aberdeen and$7347776
"""
	try:
		with open(dictionary_filepath, 'w', encoding='UTF8') as fd:
			fd.write(dictionary_data)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(dictionary_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(dictionary_filepath))

	symspell = symspellpy.SymSpell()
	# Specify "$" as the custom separator in load_dictionary().
	symspell.load_dictionary(dictionary_filepath, term_index=0, count_index=1, separator='$')

	# Print out first 5 elements to demonstrate that dictionary is successfully loaded.
	print(list(itertools.islice(symspell.words.items(), 5)))

	symspell = symspellpy.SymSpell()
	symspell.load_bigram_dictionary(dictionary_filepath, 0, 1, separator='$')

	# Print out first 5 elements to demonstrate that dictionary is successfully loaded.
	print(list(itertools.islice(symspell.bigrams.items(), 5)))

	#--------------------
	# Create dictionary.

	corpus_filepath = './corpus.txt'
	corpus_data = "abc abc-def abc_def abc'def abc qwe qwe1 1qwe q1we 1234 1234"
	try:
		with open(corpus_filepath, 'w', encoding='UTF8') as fd:
			fd.write(corpus_data)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(corpus_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(corpus_filepath))

	# Create dictionary from plain text file.
	symspell = symspellpy.SymSpell()
	symspell.create_dictionary(corpus_filepath)

	print('symspell.words =', symspell.words)

	# Build corpus.
	corpus = list()
	for line in sklearn.datasets.fetch_20newsgroups().data:
		line = line.replace('\n', ' ').replace('\t', ' ').lower()
		line = re.sub('[^a-z ]', ' ', line)
		tokens = line.split(' ')
		tokens = [token for token in tokens if len(token) > 0]
		corpus.extend(tokens)

	corpus = collections.Counter(corpus)
	#print('corpus =', corpus)

	dictionary_filepath = './dictionary.txt'
	try:
		with open(dictionary_filepath, 'w', encoding='UTF8') as fd:
			for key, value in corpus.items():
				fd.write('{} {}\n'.format(key, value))
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(dictionary_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(dictionary_filepath))

# REF [site] >> https://symspellpy.readthedocs.io/en/latest/examples/lookup.html
def lookup_example():
	dictionary_filepath = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')

	symspell = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

	# term_index is the column of the term and count_index is the column of the term frequency.
	symspell.load_dictionary(dictionary_filepath, term_index=0, count_index=1)

	input_term = 'memebers'  # Misspelling of 'members'.

	# Lookup suggestions for single-word input strings.
	#	Max edit distance per lookup (max_edit_distance_lookup <= max_dictionary_edit_distance).
	suggestions = symspell.lookup(input_term, symspellpy.Verbosity.CLOSEST, max_edit_distance=2)

	# Display suggestion term, term frequency, and edit distance.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

	#--------------------
	# Return original word if no correction within edit distance is found.
	input_term = 'apastraphee'  # Misspelling of 'apostrophe'.

	# Note that suggestions would have been empty if include_unknown was False.
	suggestions = symspell.lookup(input_term, symspellpy.Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)

	# Display suggestion term, term frequency, and edit distance.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

	#--------------------
	# Avoid correcting phrases matching regex.
	input_term = 'members1'

	# Note that "members, 1, 226656153" would be returned if ignore_token isn't specified.
	suggestions = symspell.lookup(input_term, symspellpy.Verbosity.CLOSEST, max_edit_distance=2, ignore_token=r'\w+\d')

	# Display suggestion term, term frequency, and edit distance.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

	#--------------------
	# Keep original casing.
	input_term = 'mEmEbers'

	# Note that the uppercase of the second "E" was not passed on to "b" in the corrected word.
	suggestions = symspell.lookup(input_term, symspellpy.Verbosity.CLOSEST, max_edit_distance=2, transfer_casing=True)

	# Display suggestion term, term frequency, and edit distance.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

def lookup_compound_example():
	dictionary_filepath = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')
	bigram_filepath = pkg_resources.resource_filename('symspellpy', 'frequency_bigramdictionary_en_243_342.txt')

	symspell = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

	# term_index is the column of the term and count_index is the column of the term frequency.
	symspell.load_dictionary(dictionary_filepath, term_index=0, count_index=1)
	symspell.load_bigram_dictionary(bigram_filepath, term_index=0, count_index=2)

	input_term = ("whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixtgrade and ins pired him")

	# Lookup suggestions for multi-word input strings (supports compound splitting & merging).
	#	Max edit distance per lookup (per single word, not per whole input string).
	suggestions = symspell.lookup_compound(input_term, max_edit_distance=2)

	# Display suggestion term, edit distance, and term frequency.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

	#--------------------
	# Keep original casing.
	input_term = ("whereis th elove heHAd dated forImuch of thEPast who couqdn'tread in sixtgrade and ins pired him")

	suggestions = symspell.lookup_compound(input_term, max_edit_distance=2, transfer_casing=True)

	# Display suggestion term, edit distance, and term frequency.
	if suggestions:
		for suggestion in suggestions:
			print(suggestion)
	else:
		print('No suggestion.')

# REF [site] >> https://symspellpy.readthedocs.io/en/latest/examples/word_segmentation.html
def word_segmentation_example():
	dictionary_path = pkg_resources.resource_filename('symspellpy', 'frequency_dictionary_en_82_765.txt')

	# Set max_dictionary_edit_distance to avoid spelling correction.
	symspell = symspellpy.SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
	# term_index is the column of the term and count_index is the column of the term frequency.
	symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

	# A sentence without any spaces.
	input_term = 'thequickbrownfoxjumpsoverthelazydog'
	result = symspell.word_segmentation(input_term)
	print('{}, {}, {}'.format(result.corrected_string, result.distance_sum, result.log_prob_sum))

def main():
	#dictionary_example()
	#lookup_example()
	#lookup_compound_example()
	word_segmentation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
