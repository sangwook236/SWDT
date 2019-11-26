#!/usr/bin/env python

# REF [site] >>
#	https://github.com/koalanlp/koalanlp
#	http://koalanlp.github.io/koalanlp/
#	http://koalanlp.github.io/koalanlp/usage/

# export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

from koalanlp import API
from koalanlp.proc import SentenceSplitter, Tagger

# REF [site] >> http://koalanlp.github.io/koalanlp/usage/SentenceSplit.html
def split_sentences():
	# API.KMR, API.EUNJEON, API.ARIRANG, API.RHINO, API.DAON, API.OKT, API.KKMA, API.HNN, API.ETRI.

	splitter = SentenceSplitter(splitter_type=API.HANNANUM)
	paragraph = splitter('분리할 문장을 이렇게 넣으면 문장이 분리됩니다. 간단하죠?')
	#splitter.sentences('분리할 문장을 이렇게 넣으면 문장이 분리됩니다. 간단하죠?')
	#splitter.invoke('분리할 문장을 이렇게 넣으면 문장이 분리됩니다. 간단하죠?')

	print(paragraph[0])
	print(paragraph[1])

	#--------------------
	tagger = Tagger(API.EUNJEON)  # 품사분석기.
	tagged_sentence = tagger.tagSentence('무엇인가 품사분석을 수행할 문단')
	paragraph = SentenceSplitter.sentencesTagged(tagged_sentence[0])  # tagged_sentence는 각 인자별로 한 문장으로 간주된 List[Sentence]임.
	print(paragraph)

def main():
	split_sentences()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
