#!/usr/bin/env python

# REF [site] >>
#	https://github.com/konlpy/konlpy
#	http://konlpy.org/ko/latest/
#	http://konlpy.org/ko/latest/data/
#	http://konlpy.org/ko/latest/references/

import os
from konlpy.tag import Kkma, Hannanum, Komoran, Mecab, Okt
from konlpy.corpus import kolaw, kobill
from konlpy.utils import pprint
import nltk
import wordcloud
import matplotlib.pyplot as plt

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_kkma_example():
	kkma = Kkma()
	print(kkma.tagset)

	pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
	pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
	pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))

	print(kkma.morphs(u'공부를 하면할수록 모르는게 많다는 것을 알게 됩니다.'))
	print(kkma.nouns(u'대학에서 DB, 통계학, 이산수학 등을 배웠지만...'))
	print(kkma.pos(u'다 까먹어버렸네요?ㅋㅋ'))
	print(kkma.sentences(u'그래도 계속 공부합니다. 재밌으니까!'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_hannanum_example():
	hannanum = Hannanum()
	print(hannanum.tagset)

	print(hannanum.analyze(u'롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))
	print(hannanum.morphs(u'롯데마트의 흑마늘 양념 치킨이 논란이 되고 있다.'))
	print(hannanum.nouns(u'다람쥐 헌 쳇바퀴에 타고파'))
	print(hannanum.pos(u'웃으면 더 행복합니다!'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_komoran_example():
	# REF [file] >> ${konlpy_HOME}/konlpy/data/tagset/komoran.json
	"""
	In ./user_dic.txt:
	코모란     NNP
	오픈소스    NNG
	바람과 함께 사라지다     NNP
	"""

	komoran = Komoran(userdic='./user_dic.txt')
	print(komoran.tagset)

	print(komoran.morphs(u'우왕 코모란도 오픈소스가 되었어요'))
	print(komoran.nouns(u'오픈소스에 관심 많은 멋진 개발자님들!'))
	print(komoran.pos(u'혹시 바람과 함께 사라지다 봤어?'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_mecab_example():
	# Mecab is not supported on Windows.
	mecab = Mecab()
	print(mecab.tagset)

	print(mecab.morphs(u'영등포구청역에 있는 맛집 좀 알려주세요.'))
	print(mecab.nouns(u'우리나라에는 무릎 치료를 잘하는 정형외과가 없는가!'))
	print(mecab.pos(u'자연주의 쇼핑몰은 어떤 곳인가?'))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.tag
def simple_okt_example():
	# Twitter() has changed to Okt() since v0.5.0.
	okt = Okt()
	print(okt.tagset)

	print(okt.morphs(u'단독입찰보다 복수입찰의 경우'))
	print(okt.nouns(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
	print(okt.phrases(u'날카로운 분석과 신뢰감 있는 진행으로'))
	print(okt.pos(u'이것도 되나욬ㅋㅋ'))
	print(okt.pos(u'이것도 되나욬ㅋㅋ', norm=True))
	print(okt.pos(u'이것도 되나욬ㅋㅋ', norm=True, stem=True))

# REF [site] >> http://konlpy.org/ko/latest/api/konlpy.corpus/
def simple_kolaw_corpus_example():
	fids = kolaw.fileids()
	print(fids)

	fobj = kolaw.open(fids[0])
	print(fobj.read(140))

	c = kolaw.open('constitution.txt').read()
	print(c[:10])

# REF [site] >> http://konlpy.org/ko/latest/data/
def simple_kobill_corpus_example():
	fids = kobill.fileids()
	print(fids)

	d = kobill.open('1809890.txt').read()
	print(d[:15])

# REF [site] >> https://datascienceschool.net/view-notebook/70ce46db4ced4a999c6ec349df0f4eb0/
def integration_with_nltk():
	okt = Okt()
	c = kolaw.open('constitution.txt').read()

	text = nltk.Text(okt.nouns(c), name='kolaw')
	#print(text.vocab())
	#print(len(text.vocab().items()))
	#text.vocab().plot()

	text.plot(30)
	plt.show()

# REF [site] >> https://datascienceschool.net/view-notebook/70ce46db4ced4a999c6ec349df0f4eb0/
def integration_with_wordcloud():
	if 'posix' == os.name:
		#font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
		font_path = '/usr/share/fonts/truetype/gulim.ttf'
	else:
		#font_path = 'C:/Windows/Fonts/NanumGothic.ttf'
		font_path = 'C:/Windows/Fonts/gulim.ttc'

	okt = Okt()
	c = kolaw.open('constitution.txt').read()

	text = nltk.Text(okt.nouns(c), name='kolaw')

	wc = wordcloud.WordCloud(width=1000, height=600, background_color='white', font_path=font_path)
	plt.imshow(wc.generate_from_frequencies(text.vocab()))
	plt.axis('off')
	plt.show()

def main():
	#simple_kkma_example()
	#simple_hannanum_example()
	#simple_komoran_example()
	#simple_mecab_example()  # Error.
	#simple_okt_example()

	#simple_kolaw_corpus_example()
	#simple_kobill_corpus_example()

	integration_with_nltk()
	#integration_with_wordcloud()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
