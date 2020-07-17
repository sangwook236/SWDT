#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/OpenNMT/CTranslate2

import time
import ctranslate2

def quickstart_example():
	# Convert a model trained with OpenNMT-py or OpenNMT-tf.
	#	REF [site] >> https://github.com/OpenNMT/CTranslate2

	#--------------------
	translator = ctranslate2.Translator('ende_ctranslate2/')

	print('Start translating...')
	start_time = time.time()
	# Translate tokenized inputs.
	translated = translator.translate_batch([['▁H', 'ello', '▁world', '!']])
	print('End translating: {} secs.'.format(time.time() - start_time))

	print('Tokens: {}.'.format(translated[0][0]['tokens']))
	print('Score = {}.'.format(translated[0][0]['score']))

def main():
	quickstart_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
