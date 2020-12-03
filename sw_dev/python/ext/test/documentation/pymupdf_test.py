#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/pymupdf/PyMuPDF

import fitz

# REF [site] >> https://pymupdf.readthedocs.io/en/latest/tutorial.html
def tutorial():
	pdf_filepath = '/path/to/example.pdf'

	# Open a document.
	doc = fitz.open(pdf_filepath)
	#doc = fitz.Document(pdf_filepath)

	print('#page = {}.'.format(doc.pageCount))
	print('Metadata = {}.'.format(doc.metadata))
	print('Toc:\n{}.'.format(doc.getToC()))

	# Loads page number 'pageno' of the document (0-based).
	pageno = 1
	page = doc.loadPage(pageno)
	page = doc[pageno]

	for page in doc:
	#for page in reversed(doc):
	#for page in doc.pages(start, stop, step):
		links = page.getLinks()
		annotations = page.annots()
		fields = page.widgets()
		pix = page.getPixmap()

		text = page.getText(opt='text')  # {'text', 'blocks', 'words', 'html', 'dict', 'rawdict', 'xhtml', 'xml'}.

		areas = page.searchFor('mupdf')

def save_to_image_example():
	pdf_filepath = '/path/to/infile.pdf'
	png_filepath = '/path/to/outfile.png'

	doc = fitz.open(pdf_filepath)
	page = doc.loadPage(0)
	pix = page.getPixmap()
	pix.writePNG(png_filepath)

def main():
	tutorial()

	#save_to_image_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
