#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/chrismattmann/tika-python

import tika.parser, tika.config, tika.language, tika.translate

def basic_example():
	pdf_filepath = './Performers.pdf'

	# Open a PDF file.
	parsed = tika.parser.from_file(pdf_filepath)
	#parsed = tika.parser.from_file(pdf_filepath, 'http://tika:9998/tika')
	#parsed = tika.parser.from_buffer('Good evening, Dave', 'http://tika:9998/tika')

	print('Keys = {}.'.format(parsed.keys()))

	if 'status' in parsed.keys():
		status = parsed['status']  # status returned from tika server, 200 for success.
		print('Status = {}.'.format(status))

	if 'metadata' in parsed.keys():
		metadata = parsed['metadata']
		print('Metadata = {}.'.format(metadata))

	if 'content' in parsed.keys():
		content = parsed['content']
		#text = parsed['text']

		print('Content:\n{}'.format(content))  # Type: str.

	#--------------------
	#print(tika.config.getParsers())
	#print(tika.config.getMimeTypes())
	#print(tika.config.getDetectors())

	print(tika.language.from_file(pdf_filepath))
	#print(tika.translate.from_file(pdf_filepath, 'en', 'es'))

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
