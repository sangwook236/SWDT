#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://spacy.io/

import collections, re
import numpy as np
import spacy
import spacy.matcher

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

def simple_korean_example():
	# REF [site] >> https://spacy.io/usage/models#own-models

	# spaCy does not support Korean yet.

	if True:
		from spacy.ko import Korean  # FIXME [implement] >> Korean must be implemented.
		nlp = Korean()
	elif False:
		import ko_model_name  # FIXME [implement] >> ko_model_name must be implemented.
		nlp = ko_model_name.load()
	elif False:
		# If you want to be able to load the model via spacy.load(), you'll have to create a shortcut link for it.
		# This will create a symlink in spacy/data and lets you load the model as spacy.load('ko').
		# Make model package 'ko_model_name' available as 'ko' shortcut:
		#	python -m spacy link ko_model_name ko

		nlp = spacy.load('ko')

	doc = nlp('안녕하세요')

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

# REF [site] >> https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
def intro_to_NLP_with_spaCy_example():
	# Set up spaCy.
	parser = spacy.en.English()

	# NOTE [info] >> The first time you run spaCy in a file it takes a little while to load up its modules.

	#--------------------
	# spaCy does tokenization, sentence recognition, part of speech tagging, lemmatization, dependency parsing, and named entity recognition all at once!

	# Test data.
	multiSentence = 'There is an art, it says, or rather, a knack to flying.' \
		'The knack lies in learning how to throw yourself at the ground and miss.' \
		'In the beginning the Universe was created. This has made a lot of people '\
		'very angry and been widely regarded as a bad move.'

	# Parse text.
	parsedData = parser(multiSentence)

	#--------------------
	# Let's look at the tokens:
	# All you have to do is iterate through the parsedData.
	# Each token is an object with lots of different properties.
	# A property with an underscore at the end returns the string representation
	# while a property without the underscore returns an index (int) into spaCy's vocabulary.
	# The probability estimate is based on counts from a 3 billion word corpus, smoothed using the Simple Good-Turing method.
	for i, token in enumerate(parsedData):
		print('original:', token.orth, token.orth_)
		print('lowercased:', token.lower, token.lower_)
		print('lemma:', token.lemma, token.lemma_)
		print('shape:', token.shape, token.shape_)
		print('prefix:', token.prefix, token.prefix_)
		print('suffix:', token.suffix, token.suffix_)
		print('log probability:', token.prob)
		print('Brown cluster id:', token.cluster)
		print('----------------------------------------')
		if i > 1:
			break

	#--------------------
	# Let's look at the sentences:
	sents = []
	# The "sents" property returns spans.
	# Spans have indices into the original string where each index value represents a token.
	for span in parsedData.sents:
		# Go from the start to the end of each span, returning each token in the sentence combine each token using join().
		sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
		sents.append(sent)

	for sentence in sents:
		print(sentence)

	#--------------------
	# Let's look at the part of speech tags of the first sentence:
	for span in parsedData.sents:
		sent = [parsedData[i] for i in range(span.start, span.end)]
		break

	for token in sent:
		print(token.orth_, token.pos_)

	#--------------------
	# Let's look at the dependencies of this example:
	example = 'The boy with the spotted dog quickly ran after the firetruck.'
	parsedEx = parser(example)

	# Shown as: original token, dependency tag, head word, left dependents, right dependents.
	for token in parsedEx:
		print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])

	#--------------------
	# Let's look at the named entities of this example:
	example = "Apple's stocks dropped dramatically after the death of Steve Jobs in October."
	parsedEx = parser(example)

	for token in parsedEx:
		print(token.orth_, token.ent_type_ if token.ent_type_ != '' else '(not an entity)')

	print('-------------- entities only ---------------')
	# If you just want the entities and nothing else, you can do access the parsed examples "ents" property like this:
	ents = list(parsedEx.ents)
	for entity in ents:
		print(entity.label, entity.label_, ' '.join(t.orth_ for t in entity))

	#--------------------
	# spaCy is trained to attempt to handle messy data, including emoticons and other web-based features.

	messyData = 'lol that is rly funny :) This is gr8 i rate it 8/8!!!'
	parsedData = parser(messyData)

	for token in parsedData:
		print(token.orth_, token.pos_, token.lemma_)

	#--------------------
	# spaCy has word vector representations built in!

	# You can access known words from the parser's vocabulary.
	nasa = parser.vocab['NASA']

	# Cosine similarity.
	cosine = lambda v1, v2: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

	# Gather all known words, take only the lowercased versions.
	allWords = list({w for w in parser.vocab if w.has_repvec and w.orth_.islower() and w.lower_ != 'nasa'})

	# Sort by similarity to NASA.
	allWords.sort(key=lambda w: cosine(w.repvec, nasa.repvec))
	allWords.reverse()
	print('Top 10 most similar words to NASA:')
	for word in allWords[:10]:   
		print(word.orth_)

	#--------------------
	# You can do cool things like extract Subject, Verb, Object triples from the dependency parse if you use my code in subject_object_extraction.py.
	# Note: Doesn't work on complicated sentences. Fails if the dependency parse is incorrect.

	#--------------------
	# If you want to include spaCy in your machine learning it is not too difficult.

# REF [site] >> https://realpython.com/natural-language-processing-spacy-python/
def natural_language_processing_spaCy_example():
	nlp = spacy.load('en_core_web_sm')

	#--------------------
	# How to read a string.
	introduction_text = ('This tutorial is about Natural Language Processing in Spacy.')
	introduction_doc = nlp(introduction_text)
	# Extract tokens for the given doc.
	print([token.text for token in introduction_doc])

	# How to read a text file.
	if False:
		text_filepath = './introduction.txt'
		introduction_file_text = open(text_filepath).read()
		introduction_file_doc = nlp(introduction_file_text)
		# Extract tokens for the given doc.
		print([token.text for token in introduction_file_doc])

	#--------------------
	# Sentence detection.

	print('----- Sentence detection -----')

	about_text = ('Gus Proto is a Python developer currently'
		' working for a London-based Fintech'
		' company. He is interested in learning'
		' Natural Language Processing.')
	about_doc = nlp(about_text)
	sentences = list(about_doc.sents)

	print('len(sentences) =', len(sentences))
	for sentence in sentences:
		print(sentence)

	# An ellipsis(...) is used as the delimiter.
	if False:
		def set_custom_boundaries(doc):
			# Adds support to use '...' as the delimiter for sentence detection.
			for token in doc[:-1]:
				if token.text == '...':
					doc[token.i+1].is_sent_start = True
			return doc

		ellipsis_text = ('Gus, can you, ... never mind, I forgot'
			' what I was saying. So, do you think'
			' we should ...')

		# Load a new model instance.
		custom_nlp = spacy.load('en_core_web_sm')
		custom_nlp.add_pipe(set_custom_boundaries, before='parser')

		custom_ellipsis_doc = custom_nlp(ellipsis_text)
		custom_ellipsis_sentences = list(custom_ellipsis_doc.sents)
		for sentence in custom_ellipsis_sentences:
			print(sentence)

		# Sentence detection with no customization.
		ellipsis_doc = nlp(ellipsis_text)
		ellipsis_sentences = list(ellipsis_doc.sents)
		for sentence in ellipsis_sentences:
			print(sentence)

	#--------------------
	# Tokenization.

	print('----- Tokenization -----')

	for token in about_doc:
		print(token, token.idx)

	for token in about_doc:
		print(token, token.idx, token.text_with_ws,
			token.is_alpha, token.is_punct, token.is_space,
			token.shape_, token.is_stop)

	# Customize tokenization by updating the tokenizer property on the nlp object.
	if False:
		custom_nlp = spacy.load('en_core_web_sm')
		prefix_re = spacy.util.compile_prefix_regex(custom_nlp.Defaults.prefixes)
		suffix_re = spacy.util.compile_suffix_regex(custom_nlp.Defaults.suffixes)
		infix_re = re.compile(r'''[-~]''')

		def customize_tokenizer(nlp):
			# In order for you to customize, you can pass various parameters to the Tokenizer class:
			#	nlp.vocab is a storage container for special cases and is used to handle cases like contractions and emoticons.
			#	prefix_search is the function that is used to handle preceding punctuation, such as opening parentheses.
			#	infix_finditer is the function that is used to handle non-whitespace separators, such as hyphens.
			#	suffix_search is the function that is used to handle succeeding punctuation, such as closing parentheses.
			#	token_match is an optional boolean function that is used to match strings that should never be split. It overrides the previous rules and is useful for entities like URLs or numbers.

			# Adds support to use '-' as the delimiter for tokenization.
			return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
				suffix_search=suffix_re.search,
				infix_finditer=infix_re.finditer,
				token_match=None)

		custom_nlp.tokenizer = customize_tokenizer(custom_nlp)

		custom_tokenizer_about_doc = custom_nlp(about_text)
		print([token.text for token in custom_tokenizer_about_doc])

	#--------------------
	# Stop words.
	#	Stop words are the most common words in a language.
	#	In the English language, some examples of stop words are the, are, but, and they.
	#	Most sentences need to contain stop words in order to be full sentences that make sense.

	print('----- Stop words -----')

	spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

	print('len(spacy_stopwords) =', len(spacy_stopwords))
	for stop_word in list(spacy_stopwords)[:10]:
		print(stop_word)

	for token in about_doc:
		if not token.is_stop:
			print(token)

	about_no_stopword_doc = [token for token in about_doc if not token.is_stop]
	print(about_no_stopword_doc)

	#--------------------
	# Lemmatization.
	#	Lemmatization is the process of reducing inflected forms of a word while still ensuring that the reduced form belongs to the language.
	#	This reduced form or root word is called a lemma.

	print('----- Lemmatization -----')

	conference_help_text = ('Gus is helping organize a developer'
		'conference on Applications of Natural Language'
		' Processing. He keeps organizing local Python meetups'
		' and several internal talks at his workplace.')

	conference_help_doc = nlp(conference_help_text)
	for token in conference_help_doc:
		print(token, token.lemma_)

	#--------------------
	# Word frequency.

	print('----- Word frequency -----')

	complete_text = ('Gus Proto is a Python developer currently'
		'working for a London-based Fintech company. He is'
		' interested in learning Natural Language Processing.'
		' There is a developer conference happening on 21 July'
		' 2019 in London. It is titled "Applications of Natural'
		' Language Processing". There is a helpline number '
		' available at +1-1234567891. Gus is helping organize it.'
		' He keeps organizing local Python meetups and several'
		' internal talks at his workplace. Gus is also presenting'
		' a talk. The talk will introduce the reader about "Use'
		' cases of Natural Language Processing in Fintech".'
		' Apart from his work, he is very passionate about music.'
		' Gus is learning to play the Piano. He has enrolled '
		' himself in the weekend batch of Great Piano Academy.'
		' Great Piano Academy is situated in Mayfair or the City'
		' of London and has world-class piano instructors.')

	complete_doc = nlp(complete_text)

	# Remove stop words and punctuation symbols.
	words = [token.text for token in complete_doc if not token.is_stop and not token.is_punct]
	#words = [token.text for token in complete_doc if not token.is_punct]

	# 5 commonly occurring words with their frequencies.
	word_freq = collections.Counter(words)
	common_words = word_freq.most_common(5)
	print(common_words)

	# Unique words.
	unique_words = [word for (word, freq) in word_freq.items() if freq == 1]
	print(unique_words)

	#--------------------
	# Part-of-Speech (POS) tagging.
	#	Part of speech or POS is a grammatical role that explains how a particular word is used in a sentence.
	#	There are eight parts of speech:
	#		Noun, pronoun, adjective, verb, adverb, preposition, conjunction, interjection.

	print('----- Part-of-Speech (POS) tagging -----')

	# Two attributes of the Token class are accessed:
	#	tag_ lists the fine-grained part of speech.
	#	pos_ lists the coarse-grained part of speech.
	# spacy.explain gives descriptive details about a particular POS tag.
	for token in about_doc:
		print(token, token.tag_, token.pos_, spacy.explain(token.tag_))

	nouns = []
	adjectives = []
	for token in about_doc:
		if token.pos_ == 'NOUN':
			nouns.append(token)
		if token.pos_ == 'ADJ':
			adjectives.append(token)

	#--------------------
	# Visualization: Using displaCy.
	#	You can use it to visualize a dependency parse or named entities in a browser or a Jupyter notebook.

	print('----- Visualization -----')

	about_interest_text = ('He is interested in learning Natural Language Processing.')
	about_interest_doc = nlp(about_interest_text)

	# This code will spin a simple web server.
	# You can see the visualization by opening http://127.0.0.1:5000 in your browser.
	#spacy.displacy.serve(about_interest_doc, style='dep')  # Error.

	# In a Jupyter notebook.
	#spacy.displacy.render(about_interest_doc, style='dep', jupyter=True)

	#--------------------
	# Preprocessing functions.
	#	You can create a preprocessing function that takes text as input and applies the following operations:
	#		Lowercases the text
	#		Lemmatizes each token
	#		Removes punctuation symbols
	#		Removes stop words
	#	A preprocessing function converts text to an analyzable format.
	#	It's necessary for most NLP tasks.

	print('----- Preprocessing functions -----')

	def is_token_allowed(token):
		'''Only allow valid tokens which are not stop words and punctuation symbols.
		'''
		if (not token or not token.string.strip() or token.is_stop or token.is_punct):
			return False
		return True

	def preprocess_token(token):
		# Reduce token to its lowercase lemma form.
		return token.lemma_.strip().lower()

	complete_filtered_tokens = [preprocess_token(token) for token in complete_doc if is_token_allowed(token)]
	print('complete_filtered_tokens =', complete_filtered_tokens)

	#--------------------
	# Rule-based matching.
	#	Rule-based matching is one of the steps in extracting information from unstructured text.
	#	It's used to identify and extract tokens and phrases according to patterns (such as lowercase) and grammatical features (such as part of speech).
	#	Rule-based matching can use regular expressions to extract entities (such as phone numbers) from an unstructured text.
	#	It's different from extracting text using regular expressions only in the sense that regular expressions don't consider the lexical and grammatical attributes of the text.

	print('----- Rule-based matching -----')

	matcher = spacy.matcher.Matcher(nlp.vocab)
	def extract_full_name(nlp_doc):
		pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
		matcher.add('FULL_NAME', None, pattern)
		matches = matcher(nlp_doc)
		for match_id, start, end in matches:
			span = nlp_doc[start:end]
			return span.text

	print('extract_full_name(about_doc) =', extract_full_name(about_doc))

	conference_org_text = ('There is a developer conference'
		'happening on 21 July 2019 in London. It is titled'
		' "Applications of Natural Language Processing".'
		' There is a helpline number available'
		' at (123) 456-789')

	matcher = spacy.matcher.Matcher(nlp.vocab)
	def extract_phone_number(nlp_doc):
		pattern = [{'ORTH': '('}, {'SHAPE': 'ddd'},
				   {'ORTH': ')'}, {'SHAPE': 'ddd'},
				   {'ORTH': '-', 'OP': '?'},
				   {'SHAPE': 'ddd'}]
		matcher.add('PHONE_NUMBER', None, pattern)
		matches = matcher(nlp_doc)
		for match_id, start, end in matches:
			span = nlp_doc[start:end]
			return span.text

	conference_org_doc = nlp(conference_org_text)

	print('extract_phone_number(conference_org_doc) =', extract_phone_number(conference_org_doc))

	#--------------------
	# Dependency parsing.
	#	Dependency parsing is the process of extracting the dependency parse of a sentence to represent its grammatical structure.
	#	It defines the dependency relationship between headwords and their dependents.
	#	The head of a sentence has no dependency and is called the root of the sentence.
	#	The verb is usually the head of the sentence.
	#	All other words are linked to the headword.
	#	The dependencies can be mapped in a directed graph representation:
	#		Words are the nodes.
	#		The grammatical relationships are the edges.
	#	Dependency parsing helps you know what role a word plays in the text and how different words relate to each other.
	#	It's also used in shallow parsing and named entity recognition.

	print('----- Dependency parsing -----')

	piano_text = 'Gus is learning piano'
	piano_doc = nlp(piano_text)

	# In this example, the sentence contains three relationships:
	#	nsubj is the subject of the word. Its headword is a verb.
	#	aux is an auxiliary word. Its headword is a verb.
	#	dobj is the direct object of the verb. Its headword is a verb.
	for token in piano_doc:
		print (token.text, token.tag_, token.head.text, token.dep_)

	# Use displaCy to visualize the dependency tree.
	# This code will produce a visualization that can be accessed by opening http://127.0.0.1:5000 in your browser.
	#spacy.displacy.serve(piano_doc, style='dep')

	#--------------------
	# Navigating the tree and subtree.
	#	The dependency parse tree has all the properties of a tree.
	#	This tree contains information about sentence structure and grammar and can be traversed in different ways to extract relationships.
	#	spaCy provides attributes like children, lefts, rights, and subtree to navigate the parse tree.

	print('----- Navigating the tree and subtree -----')

	one_line_about_text = ('Gus Proto is a Python developer currently working for a London-based Fintech company')
	one_line_about_doc = nlp(one_line_about_text)

	# Extract children of 'developer'.
	print([token.text for token in one_line_about_doc[5].children])
	# Extract previous neighboring node of 'developer'.
	print (one_line_about_doc[5].nbor(-1))
	# Extract next neighboring node of 'developer'.
	print (one_line_about_doc[5].nbor())
	# Extract all tokens on the left of 'developer'.
	print([token.text for token in one_line_about_doc[5].lefts])
	# Extract tokens on the right of 'developer'.
	print([token.text for token in one_line_about_doc[5].rights])
	# Print subtree of 'developer'.
	print (list(one_line_about_doc[5].subtree))

	# Construct a function that takes a subtree as an argument and returns a string by merging words in it.
	def flatten_tree(tree):
		return ''.join([token.text_with_ws for token in list(tree)]).strip()

	# Print flattened subtree of 'developer'.
	print('flatten_tree(one_line_about_doc[5].subtree) =', flatten_tree(one_line_about_doc[5].subtree))

	#--------------------
	# Shallow parsing.
	#	Shallow parsing, or chunking, is the process of extracting phrases from unstructured text.
	#	Chunking groups adjacent tokens into phrases on the basis of their POS tags.
	#	There are some standard well-known chunks such as noun phrases, verb phrases, and prepositional phrases.

	print('----- Shallow parsing -----')

	# Noun phrase detection.
	#	A noun phrase is a phrase that has a noun as its head.
	#	It could also include other kinds of words, such as adjectives, ordinals, determiners.
	#	Noun phrases are useful for explaining the context of the sentence.
	#	They help you infer what is being talked about in the sentence.

	conference_text = ('There is a developer conference happening on 21 July 2019 in London.')
	conference_doc = nlp(conference_text)

	# Extract noun phrases.
	for chunk in conference_doc.noun_chunks:
		print(chunk)

	# Verb phrase detection.
	#	A verb phrase is a syntactic unit composed of at least one verb.
	#	This verb can be followed by other chunks, such as noun phrases.
	#	Verb phrases are useful for understanding the actions that nouns are involved in.
	#	spaCy has no built-in functionality to extract verb phrases, so you'll need a library called 'textacy'.

	if False:
		import textacy

		about_talk_text = ('The talk will introduce reader about Use cases of Natural Language Processing in Fintech')
		about_talk_doc = textacy.make_spacy_doc(about_talk_text, lang='en_core_web_sm')

		pattern = r'(<VERB>?<ADV>*<VERB>+)'
		verb_phrases = textacy.extract.pos_regex_matches(about_talk_doc, pattern)

		# Print all verb phrases.
		for chunk in verb_phrases:
			print(chunk.text)

		# Extract noun phrase to explain what nouns are involved.
		for chunk in about_talk_doc.noun_chunks:
			print(chunk)

	#--------------------
	# Named entity recognition (NER).
	#	Named Entity Recognition (NER) is the process of locating named entities in unstructured text and then classifying them into pre-defined categories, such as person names, organizations, locations, monetary values, percentages, time expressions, and so on.
	#	You can use NER to know more about the meaning of your text.
	#	For example, you could use it to populate tags for a set of documents in order to improve the keyword search.
	#	You could also use it to categorize customer support tickets into relevant categories.

	print('----- Named entity recognition -----')

	piano_class_text = ('Great Piano Academy is situated in Mayfair or the City of London and has world-class piano instructors.')
	piano_class_doc = nlp(piano_class_text)

	# In the example, ent is a Span object with various attributes:
	#	text gives the Unicode text representation of the entity.
	#	start_char denotes the character offset for the start of the entity.
	#	end_char denotes the character offset for the end of the entity.
	#	label_ gives the label of the entity.
	# spacy.explain gives descriptive details about an entity label.
	# The spaCy model has a pre-trained list of entity classes.

	for ent in piano_class_doc.ents:
		print(ent.text, ent.start_char, ent.end_char, ent.label_, spacy.explain(ent.label_))

	# You can use displaCy to visualize these entities.
	# If you open http://127.0.0.1:5000 in your browser, then you can see the visualization.
	#spacy.displacy.serve(piano_class_doc, style='ent')

	# You can use NER to redact people.s names from a text.
	# For example, you might want to do this in order to hide personal information collected in a survey.

	survey_text = ('Out of 5 people surveyed, James Robert, Julie Fuller and Benjamin Brooks like apples. Kelly Cox and Matthew Evans like oranges.')

	def replace_person_names(token):
		if token.ent_iob != 0 and token.ent_type_ == 'PERSON':
			return '[REDACTED] '
		return token.string

	def redact_names(nlp_doc):
		for ent in nlp_doc.ents:
			ent.merge()
		tokens = map(replace_person_names, nlp_doc)
		return ''.join(tokens)

	survey_doc = nlp(survey_text)
	print('redact_names(survey_doc) =', redact_names(survey_doc))

def main():
	# Download model.
	#	python -m spacy download en_core_web_sm

	#spacy.prefer_gpu()

	#simple_example()
	#simple_korean_example()  # Not yet completed.

	#linguistic_features_example()
	#processing_pipelines_example()

	#intro_to_NLP_with_spaCy_example()
	natural_language_processing_spaCy_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
