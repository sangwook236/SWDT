#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://spacy.io/

import spacy

def simple_example():
	# Load English tokenizer, tagger, parser, NER and word vectors.
	nlp = spacy.load('en_core_web_sm')

	# Process whole documents.
	text = ('When Sebastian Thrun started working on self-driving cars at '
			'Google in 2007, few people outside of the company took him '
			'seriously. "I can tell you very senior CEOs of major American '
			"car companies would shake my hand and turn away because I wasn't "
			'worth talking to," said Thrun, in an interview with Recode earlier '
			'this week.')
	doc = nlp(text)

	# Analyze syntax.
	print('Noun phrases:', [chunk.text for chunk in doc.noun_chunks])
	print('Verbs:', [token.lemma_ for token in doc if token.pos_ == 'VERB'])

	# Find named entities, phrases and concepts.
	for entity in doc.ents:
		print(entity.text, entity.label_)

	for sent in doc.sents:
		print(sent)

# REF [site] >> https://spacy.io/usage/linguistic-features
def linguistic_features_example():
	nlp = spacy.load('en_core_web_sm')

	# Part-of-speech tagging.
	doc = nlp('Apple is looking at buying U.K. startup for $1 billion')

	for token in doc:
		print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

	# Noun chunks.
	doc = nlp('Autonomous cars shift insurance liability toward manufacturers')

	for chunk in doc.noun_chunks:
		print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)

	# Navigating the parse tree.
	for token in doc:
		print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])

	#---------------
	# Because the syntactic relations form a tree, every word has exactly one head.
	# You can therefore iterate over the arcs in the tree by iterating over the words in the sentence.
	# This is usually the best way to match an arc of interest.

	# Finding a verb with a subject from below - good.
	verbs = set()
	for possible_subject in doc:
		if possible_subject.dep == spacy.symbols.nsubj and possible_subject.head.pos == spacy.symbols.VERB:
			verbs.add(possible_subject.head)
	print(verbs)

	# If you try to match from above, you’ll have to iterate twice.
	# Once for the head, and then again through the children.

	# Finding a verb with a subject from above - less good.
	verbs = []
	for possible_verb in doc:
		if possible_verb.pos == spacy.symbols.VERB:
			for possible_subject in possible_verb.children:
				if possible_subject.dep == spacy.symbols.nsubj:
					verbs.append(possible_verb)
					break

# REF [site] >> https://spacy.io/usage/processing-pipelines
def processing_pipelines_example():
	texts = [
		'Net income was $9.4 million compared to the prior year of $2.7 million.',
		'Revenue exceeded twelve billion dollars, with a loss of $1b.',
	]

	nlp = spacy.load('en_core_web_sm')

	for doc in nlp.pipe(texts, disable=['tagger', 'parser']):
		# Do something with the doc here.
		print([(ent.text, ent.label_) for ent in doc.ents])

def main():
	# Download model.
	#	python -m spacy download en_core_web_sm

	#spacy.prefer_gpu()

	#simple_example()
	#linguistic_features_example()
	processing_pipelines_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
