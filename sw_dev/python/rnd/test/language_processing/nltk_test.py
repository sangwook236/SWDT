#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://www.nltk.org/

import nltk
from collections import defaultdict

# REF [site] >>
#	https://pythonspot.com/category/nltk/
#	https://data-flair.training/blogs/nltk-python-tutorial/
def simple_example():
	# Tokenization.
	sentence = """At eight o'clock on Thursday morning
Arthur didn't feel very good.
"""
	tokens = nltk.word_tokenize(sentence)
	print('Words = {}.'.format(tokens))

	#paragraph = 'Good morning Dr. Adams. The patient is waiting for you in room number 3.'
	paragraph = 'Mr. John Johnson Jr. was born in the U.S.A but earned his Ph.D. in Israel before joining Nike Inc. as an engineer. He also worked at craigslist.org as a business analyst.'
	sents = nltk.sent_tokenize(paragraph)
	#sents = nltk.sent_tokenize(paragraph, language='english')  # nltk.tokenize.PunktSentenceTokenizer.
	print('Sentences = {}.'.format(sents))

	#--------------------
	# Stop words.
	paragraph = 'All work and no play makes jack dull boy. All work and no play makes jack a dull boy.'
	stop_words = set(nltk.corpus.stopwords.words('english'))
	#print('English stop words = {}.'.format(stop_words))

	words = nltk.word_tokenize(paragraph)
	words_filtered = list(w for w in words if w not in stop_words)
	
	print('Words w/ stop words = {}.'.format(words))
	print('Words w/o stop words = {}.'.format(words_filtered))

	#--------------------
	# Stemming.
	#words = ['game', 'gaming', 'gamed', 'games']
	sentence = 'gaming, the gamers play games'
	words = nltk.word_tokenize(sentence)

	stemmer = nltk.stem.PorterStemmer()
	print('Stemming: {}.'.format([(word, stemmer.stem(word)) for word in words]))

	print('nltk.stem.SnowballStemmer.languages = {}.'.format(nltk.stem.SnowballStemmer.languages))
	rom_stemmer = nltk.stem.SnowballStemmer('romanian')
	print("rom_stemmer.stem('englezească') = {}.".format(rom_stemmer.stem('englezească')))

	#--------------------
	# Lemmatization.
	lemmatizer = nltk.stem.WordNetLemmatizer()

	print('Lemmatization: this -> {}.'.format(lemmatizer.lemmatize('this')))
	print('Lemmatization: believes (n) -> {}.'.format(lemmatizer.lemmatize('believes')))
	print('Lemmatization: believes (v) -> {}.'.format(lemmatizer.lemmatize('believes', pos='v')))
	print('Lemmatization: crossing (a) -> {}.'.format(lemmatizer.lemmatize('crossing', pos='a')))  # Adjective.
	print('Lemmatization: crossing (v) -> {}.'.format(lemmatizer.lemmatize('crossing', pos='v')))  # Verb.
	print('Lemmatization: crossing (n) -> {}.'.format(lemmatizer.lemmatize('crossing', pos='n')))  # Noun.
	print('Lemmatization: crossing (r) -> {}.'.format(lemmatizer.lemmatize('crossing', pos='r')))  # Adverb.

	#--------------------
	# POS tagging.

	# POS tags:
	# 	https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
	#	https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
	#nltk.help.upenn_tagset('RB')

	tagged = nltk.pos_tag(tokens)
	#print('POS tags = {}.'.format(tagged[0:6]))
	print('POS tags = {}.'.format(tagged))

	#document = 'I am a human being, capable of doing terrible things'
	document = "Whether you're new to programming or an experienced developer, it's easy to learn and use Python."
	#document = "Today the Netherlands celebrates King's Day. To honor this tradition, the Dutch embassy in San Francisco invited me to"
	sentences = nltk.sent_tokenize(document)
	for sent in sentences:
		words_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
		print('POS tagging: {}.'.format(words_tagged))
		print('Words w/ NNP tag = {}.'.format(list(word for word in words_tagged if 'NNP' in word[1])))

	#--------------------
	# Named entity.
	entities = nltk.chunk.ne_chunk(tagged)
	print('Named entities = {}.'.format(entities))

	#--------------------
	# Parse tree.
	t = nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0]
	#t.draw()

	#--------------------
	# Synonym and antonym.
	syns = nltk.corpus.wordnet.synsets('love')

	print('Synonym set of love = {}.'.format(syns))
	for syn in syns:
		print('Synonym of love = {}.'.format(syn))
		print('\tDefinition: {}.'.format(syn.definition()))
		print('\tExamples: {}.'.format(syn.examples()))
		print('\tLemmas: {}.'.format(syn.lemmas()))

	syns = nltk.corpus.wordnet.synsets('depressed')
	#syns = nltk.corpus.wordnet.synsets('beautiful')
	for syn in syns:
		print('Antonyms = {}.'.format(list(l.antonyms()[0].name() for l in syn.lemmas() if l.antonyms())))

# REF [site] >> https://pythonspot.com/category/nltk/
def training_and_prediction_example():
	def gender_features(word): 
		return {'last_letter': word[-1]} 

	# Load data.
	#names = ([(name, 'male') for name in nltk.corpus.names.words('male.txt')] + 
	#	[(name, 'female') for name in nltk.corpus.names.words('female.txt')])
	names = [
		(u'Aaron', 'male'), (u'Abbey', 'male'), (u'Abbie', 'male'),
		(u'Zorana', 'female'), (u'Zorina', 'female'), (u'Zorine', 'female')
	]

	# Train.
	featuresets = [(gender_features(n), g) for (n, g) in names] 
	train_set = featuresets
	classifier = nltk.NaiveBayesClassifier.train(train_set) 

	# Predict.
	print('Prediction = {}.'.format(classifier.classify(gender_features('Frank'))))

# REF [site] >> https://pythonspot.com/category/nltk/
def sentiment_analysis_example():
	def word_feats(words):
		return dict([(word, True) for word in words])

	positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
	negative_vocab = ['bad', 'terrible','useless', 'hate', ':(']
	neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']

	positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
	negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
	neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

	train_set = negative_features + positive_features + neutral_features

	# Train.
	classifier = nltk.classify.NaiveBayesClassifier.train(train_set) 

	# Predict.
	num_pos, num_neg = 0, 0
	sentence = 'Awesome movie, I liked it'
	sentence = sentence.lower()
	words = sentence.split(' ')
	for word in words:
		classified = classifier.classify( word_feats(word))
		if classified == 'neg':
			num_neg += 1
		if classified == 'pos':
			num_pos += 1

	print('Positive: {}.'.format(float(num_pos) / len(words)))
	print('Negative: {}.'.format(float(num_neg) / len(words)))

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
	#nltk.download()
	nltk.download('punkt')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('maxent_ne_chunker')
	nltk.download('words')
	nltk.download('treebank')
	nltk.download('stopwords')
	nltk.download('tagsets')

	#--------------------
	simple_example()

	#training_and_prediction_example()
	#sentiment_analysis_example()

	#--------------------
	#n_gram_example()
	# REF [function] >> extract_bigram_or_trigram_with_nltk() in konlpy_test.py.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
