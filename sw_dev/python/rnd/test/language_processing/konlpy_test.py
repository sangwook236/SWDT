#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/konlpy/konlpy
#	http://konlpy.org/ko/latest/
#	http://konlpy.org/ko/latest/data/
#	http://konlpy.org/ko/latest/references/

import os
import konlpy
import nltk
import wordcloud
import matplotlib.pyplot as plt

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_kkma_example():
	kkma = konlpy.tag.Kkma()

	print('kkma.tagset =', kkma.tagset)

	konlpy.utils.pprint(kkma.sentences('네, 안녕하세요. 반갑습니다.'))
	konlpy.utils.pprint(kkma.nouns('질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
	konlpy.utils.pprint(kkma.pos('오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))
	konlpy.utils.pprint(kkma.pos('오루보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))  # A typo exists.

	print(kkma.sentences('그래도 계속 공부합니다. 재밌으니까!'))
	print(kkma.nouns('대학에서 DB, 통계학, 이산수학 등을 배웠지만...'))
	print(kkma.morphs('공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))
	print(kkma.pos('다 까먹어버렸네요?ㅋㅋ'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_hannanum_example():
	hannanum = konlpy.tag.Hannanum()

	print('hannanum.tagset =', hannanum.tagset)

	print(hannanum.nouns('다람쥐 헌 쳇바퀴에 타고파'))
	print(hannanum.analyze('롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))
	print(hannanum.morphs('롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))
	print(hannanum.pos('웃으면 더 행복합니다!'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_komoran_example():
	# REF [file] >> ${konlpy_HOME}/konlpy/data/tagset/komoran.json
	"""
	In user_dic.txt:
		코모란     NNP
		오픈소스    NNG
		바람과 함께 사라지다     NNP
	"""

	komoran = konlpy.tag.Komoran(userdic='./user_dic.txt')

	print('komoran.tagset =', komoran.tagset)

	print(komoran.nouns('오픈소스에 관심 많은 멋진 개발자님들!'))
	print(komoran.morphs('우왕 코모란도 오픈소스가 되었어요'))
	print(komoran.pos('혹시 바람과 함께 사라지다 봤어?'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_mecab_example():
	# Mecab is not supported on Windows.
	mecab = konlpy.tag.Mecab()

	print('mecab.tagset =', mecab.tagset)

	print(mecab.nouns('우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!'))
	print(mecab.morphs('영등포구청역에 있는 맛집 좀 알려주세요.'))
	print(mecab.pos('자연주의 쇼핑몰은 어떤 곳인가?'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_okt_example():
	# Twitter() has changed to Okt() since v0.5.0.
	okt = konlpy.tag.Okt()

	print('okt.tagset =', okt.tagset)

	print(okt.phrases('날카로운 분석과 신뢰감 있는 진행으로'))
	print(okt.nouns('유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
	print(okt.morphs('단독입찰보다 복수입찰의 경우'))
	print(okt.pos('이것도 되나욬ㅋㅋ'))
	print(okt.pos('이것도 되나욬ㅋㅋ', norm=True))
	print(okt.pos('이것도 되나욬ㅋㅋ', norm=True, stem=True))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.corpus/
def simple_kolaw_corpus_example():
	fids = konlpy.corpus.kolaw.fileids()
	print(fids)

	fobj = konlpy.corpus.kolaw.open(fids[0])
	print(fobj.read(140))

	c = konlpy.corpus.kolaw.open('constitution.txt').read()
	print(c[:10])

# REF [site] >> http://konlpy.org/ko/latest/data/
def simple_kobill_corpus_example():
	fids = konlpy.corpus.kobill.fileids()
	print(fids)

	d = konlpy.corpus.kobill.open('1809890.txt').read()
	print(d[:15])

# REF [site] >> https://datascienceschool.net/view-notebook/70ce46db4ced4a999c6ec349df0f4eb0/
def integrate_with_nltk():
	okt = konlpy.tag.Okt()
	c = konlpy.corpus.kolaw.open('constitution.txt').read()

	text = nltk.Text(okt.nouns(c), name='kolaw')
	#print(text.vocab())
	#print(len(text.vocab().items()))
	#text.vocab().plot()

	text.plot(30)
	plt.show()

# REF [doc] >> "Python 환경에서 한글 형태소 분석기 패키지 KoNLPy 사용법.pdf"
def extract_bigram_or_trigram_with_nltk():
	bigram_measures = nltk.collocations.BigramAssocMeasures()
	doc = konlpy.corpus.kolaw.open('constitution.txt').read()
	pos = konlpy.tag.Kkma().pos(doc)
	words = [s for s, t in pos]
	tags = [t for s, t in pos]

	print('\nCollocations among tagged words:')
	finder = nltk.collocations.BigramCollocationFinder.from_words(pos)
	konlpy.utils.pprint(finder.nbest(bigram_measures.pmi, 10))  # Top 10 n-grams with highest PMI.

	print('\nCollocations among words:')
	ignored_words = ['안녕']
	finder = nltk.collocations.BigramCollocationFinder.from_words(words)
	finder.apply_word_filter(lambda w: len(w) < 2 or w in ignored_words)
	finder.apply_freq_filter(3)  # Only bigrams that appear 3+ times.
	konlpy.utils.pprint(finder.nbest(bigram_measures.pmi, 10))

	print('\nCollocations among tags:')
	finder = nltk.collocations.BigramCollocationFinder.from_words(tags)
	konlpy.utils.pprint(finder.nbest(bigram_measures.pmi, 5))

# REF [site] >> https://datascienceschool.net/view-notebook/70ce46db4ced4a999c6ec349df0f4eb0/
def integrate_with_wordcloud():
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	font_filepath = font_base_dir_path + '/kor/gulim.ttf'

	okt = konlpy.tag.Okt()
	c = konlpy.corpus.kolaw.open('constitution.txt').read()

	text = nltk.Text(okt.nouns(c), name='kolaw')

	wc = wordcloud.WordCloud(width=1000, height=600, background_color='white', font_path=font_filepath)
	plt.imshow(wc.generate_from_frequencies(text.vocab()))
	plt.axis('off')
	plt.show()

def main():
	# Initialize the Java virtual machine (JVM).
	#konlpy.jvm.init_jvm(jvmpath=None, max_heap_size=1024)	

	#--------------------
	#simple_kkma_example()
	#simple_hannanum_example()
	#simple_komoran_example()
	#simple_mecab_example()  # Error.
	#simple_okt_example()

	#simple_kolaw_corpus_example()
	#simple_kobill_corpus_example()

	#--------------------
	#integrate_with_nltk()
	extract_bigram_or_trigram_with_nltk()

	#integrate_with_wordcloud()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
