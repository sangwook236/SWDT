#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://www.nltk.org/

import nltk
from nltk.corpus import treebank

def simple_example():
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('maxent_ne_chunker')
	nltk.download('words')
	nltk.download('treebank')

	sentence = """
		At eight o'clock on Thursday morning
		Arthur didn't feel very good.
		"""
	tokens = nltk.word_tokenize(sentence)
	print(tokens)

	tagged = nltk.pos_tag(tokens)
	print(tagged[0:6])

	# Identify named entities.
	entities = nltk.chunk.ne_chunk(tagged)
	print(entities)

	# Display a parse tree.
	t = treebank.parsed_sents('wsj_0001.mrg')[0]
	t.draw()

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
