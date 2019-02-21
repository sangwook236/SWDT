#!/usr/bin/env python

# REF [site] >>
#	http://konlpy.org/ko/latest/
#	https://github.com/konlpy/konlpy

import numpy as np
from konlpy.tag import Kkma
from konlpy.utils import pprint

def simple_example():
	kkma = Kkma()
	pprint(kkma.sentences(u'네, 안녕하세요. 반갑습니다.'))
	pprint(kkma.nouns(u'질문이나 건의사항은 깃헙 이슈 트래커에 남겨주세요.'))
	pprint(kkma.pos(u'오류보고는 실행환경, 에러메세지와함께 설명을 최대한상세히!^^'))

def main():
	simple_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
