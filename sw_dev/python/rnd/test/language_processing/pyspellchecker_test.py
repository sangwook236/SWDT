#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/barrust/pyspellchecker
#	http://norvig.com/spell-correct.html

import spellchecker

# REF [site] >> https://github.com/barrust/pyspellchecker
def simple_example():
	spell = spellchecker.SpellChecker(language='en',
		local_dictionary=None,
		distance=2,
		tokenizer=None,
		case_sensitive=False)

	# Find those words that may be misspelled.
	misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

	for word in misspelled:
		# Get the one 'most likely' answer.
		print('spell.correction(word) =', spell.correction(word))

		# Get a list of 'likely' options.
		print('spell.candidates(word) =', spell.candidates(word))

	print("spell.word_probability('here') =", spell.word_probability('here'))

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
	dictionary_filepath ='./my_dictionary.json'
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
	print("spell.known(['microsoft', 'google', 'facebook']) =", spell.known(['microsoft', 'google', 'facebook']))  # Will return both now!

	print('len(spell.word_frequency.dictionary) =', len(spell.word_frequency.dictionary))
	print('spell.word_frequency.total_words =', spell.word_frequency.total_words)
	print('spell.word_frequency.unique_words =', spell.word_frequency.unique_words)
	print('len(spell.word_frequency.letters) =', len(spell.word_frequency.letters))
	print('spell.word_frequency.longest_word_length =', spell.word_frequency.longest_word_length)

	print('spell.word_frequency.tokenize(text_data)) =', list(spell.word_frequency.tokenize(text_data)))

	print('spell.word_frequency.keys()) =', list(word for idx, word in enumerate(spell.word_frequency.keys()) if idx < 20))
	print('spell.word_frequency.words()) =', list(word for idx, word in enumerate(spell.word_frequency.words()) if idx < 20))
	print('spell.word_frequency.items()) =', list(word for idx, word in enumerate(spell.word_frequency.items()) if idx < 20))

	#--------------------
	# If the words that you wish to check are long, it is recommended to reduce the distance to 1.

	spell = spellchecker.SpellChecker(distance=1)  # Set at initialization.

	# Do some work on longer words.

	spell.distance = 2  # Set the distance parameter back to the default.

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
