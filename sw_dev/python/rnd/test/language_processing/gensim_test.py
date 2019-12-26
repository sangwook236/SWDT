#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://radimrehurek.com/gensim/
#	https://github.com/RaRe-Technologies/gensim

from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

# REF [site] >> http://younggyunhahm.blogspot.com/2017/06/word-embedding-for-korean-using-gensim.html
def simple_word_vector_example():
	#with open('./example.txt', 'r', encoding='utf-8') as fd:
	#	text = fd.readlines()
	text = ['나는 너를 사랑해.\n', '나는 너를 미워해.\n']
	token = [s.split() for s in text]

	embedding = word2vec.Word2Vec(token, size=5, window=1, negative=3, min_count=1)

	if True:
		# Saves a model.
		embedding.save('./my.model')
		# Loads a model.
		model = word2vec.Word2Vec.load('./my.model')
	else:
		# word2vec file format.
		# Saves a model.
		embedding.wv.save_word2vec_format('./my.embedding', binary=False)
		# Loads a model.
		model = KeyedVectors.load_word2vec_format('./my.embedding', binary=False, encoding='utf-8')

	# Word vector.
	#print(model.wv['너를'])
	print('Word vector:', model['너를'])
	# Find the top-N most similar words.
	print('Most similar words:', model.most_similar('너를'))

	# 한국어의 경우 띄어쓰기 단위가 어절 형태이기 때문에 형태소 단위로 tokenization이 필요하며, 또한 품사태그를 포함한 word embedding 생성에서 좋은 성능을 보인다고 알려져 있다.
	# 가장 쉽게 하는 방법은 KoNLPy 를 사용하는 것: 여기
	# 한마디로 정리하면, token 리스트를 다음과 같이 구성하면 된다
	# [['나/NP', '는/JX', '너/NP', '를/JKO', '......], [..], ...]

def main():
	simple_word_vector_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
