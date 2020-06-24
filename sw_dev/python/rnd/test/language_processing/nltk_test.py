#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://www.nltk.org/

import nltk
from collections import defaultdict

def simple_example():
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('maxent_ne_chunker')
	nltk.download('words')
	nltk.download('treebank')

	sentence = """At eight o'clock on Thursday morning
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
	t = nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0]
	t.draw()

# REF [site] >> https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d
def n_gram_example():
	nltk.download('punkt')
	#nltk.download('averaged_perceptron_tagger')
	#nltk.download('maxent_ne_chunker')
	#nltk.download('words')
	nltk.download('reuters')

	# Create a placeholder for model.
	model = defaultdict(lambda: defaultdict(lambda: 0))

	# Count frequency of co-occurance  .
	for sentence in nltk.corpus.reuters.sents():
		#for w1, w2 in nltk.bigrams(sentence, pad_right=True, pad_left=True):
		#	#model[(w1,)][w2] += 1  # NOTE [info] >> Not good.
		#	model[w1][w2] += 1
		for grams in nltk.trigrams(sentence, pad_right=True, pad_left=True):
		#for grams in nltk.ngrams(sentence, n=6, pad_right=True, pad_left=True):
			model[grams[:-1]][grams[-1]] += 1

	# Transform the counts to probabilities.
	for grams in model:
		total_count = float(sum(model[grams].values()))
		for ww in model[grams]:
			model[grams][ww] /= total_count

	#--------------------
	# Predict the next word.

	# Words which start with two simple words, 'today the'.
	print("model['today'] =", dict(model['today']))
	print("model['today', 'the'] =", dict(model['today', 'the']))

	# Words which start with two simple words, 'the price'.
	print("model['the'] =", dict(model['the']))
	print("model['the', 'price'] =", dict(model['the', 'price']))

	#--------------------
	# Generate a random piece of text using our n-gram model.

	# Starting words.
	text = ['today', 'the']
	sentence_finished = False

	import random
	while not sentence_finished:
		# Select a random probability threshold.
		r = random.random()
		accumulator = 0.0

		for word in model[tuple(text[-2:])].keys():
			accumulator += model[tuple(text[-2:])][word]
			# Select words that are above the probability threshold.
			if accumulator >= r:
				text.append(word)
				break

		if text[-2:] == [None, None]:
			sentence_finished = True

	print ('Generated text =', ' '.join([t for t in text if t]))

def main():
	#simple_example()

	n_gram_example()
	# REF [function] >> extract_bigram_or_trigram_with_nltk() in konlpy_test.py.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
