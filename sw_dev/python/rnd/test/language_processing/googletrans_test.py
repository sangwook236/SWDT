#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/ssut/py-googletrans

import googletrans

def simple_example():
	translator = googletrans.Translator()

	txt = 'I would like to think while walking.'
	translated1 = translator.translate(txt, dest='it').text
	translated2 = translator.translate(translated1, dest='en').text
	print('{} -it-> {} -en-> {}'.format(txt, translated1, translated2))

	translations = translator.translate(['The quick brown fox', 'jumps over', 'the lazy dog'], dest='ko')
	for translation in translations:
		print('{} -ko-> {}'.format(translation.origin, translation.text))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
