#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/chrismattmann/tika-python

#import tika
#tika.initVM()
import tika.parser, tika.config, tika.unpack, tika.detector, tika.language, tika.translate

def basic_example():
	pdf_filepath = '/path/to/sample.pdf'

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
	# Specify output format To XHTML.
	parsed = tika.parser.from_file(pdf_filepath, xmlContent=True)
	print('parsed["metadata"] = {}.'.format(parsed['metadata']))
	print('parsed["content"] = {}.'.format(parsed['content']))

	#--------------------
	# Unpack interface.
	#	The unpack interface handles both metadata and text extraction in a single call and internally returns back a tarball of metadata and text entries that is internally unpacked, reducing the wire load for extraction.
	parsed = tika.unpack.from_file(pdf_filepath)

	# Detect interface.
	#	The detect interface provides a IANA MIME type classification.
	print('tika.detector.from_file(pdf_filepath) = {}.'.format(tika.detector.from_file(pdf_filepath)))

	# Config interface.
	print('tika.config.getParsers() = {}.'.format(tika.config.getParsers()))
	print('tika.config.getMimeTypes() = {}.'.format(tika.config.getMimeTypes()))
	print('tika.config.getDetectors() = {}.'.format(tika.config.getDetectors()))

	# Language detection interface.
	print('tika.language.from_file(pdf_filepath) = {}.'.format(tika.language.from_file(pdf_filepath)))

	# Translate interface.
	#print('tika.translate.from_file(pdf_filepath, "en", "es") = {}.'.format(tika.translate.from_file(pdf_filepath, 'en', 'es')))

# REF [site] >> https://stackoverflow.com/questions/53093531/python-apache-tika-single-page-parser
def single_page_parsing():
	pdf_filepath = '/path/to/sample.pdf'

	raw_xml = tika.parser.from_file(pdf_filepath, xmlContent=True)

	xml_body = raw_xml['content'].split('<body>')[1].split('</body>')[0]
	xml_body_without_tag = xml_body.replace('<p>', '').replace('</p>', '').replace('<div>', '').replace('</div>', '').replace('<p />', '')
	pages = xml_body_without_tag.split('<div class="page">')[1:]

	num_pages = len(pages)
	if num_pages == int(raw_xml['metadata']['xmpTPg:NPages']):
		for idx, page in enumerate(pages):
			print('------------------------------ Page {} in {}.'.format(idx, pdf_filepath))
			print(page)
	else:
		print('No page.')

def main():
	#tika.TikaClientOnly = True

	basic_example()
	#single_page_parsing()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
