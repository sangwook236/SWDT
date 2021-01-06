#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/euske/pdfminer
#	https://github.com/pdfminer/pdfminer.six

from pdfminer.pdfparser import PDFParser, PDFSyntaxError
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFEncryptionError
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator, TextConverter, XMLConverter, HTMLConverter
from pdfminer.pdftypes import resolve1, resolve_all, PDFException

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def basic_usage():
	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create a PDF resource manager object that stores shared resources.
		rsrcmgr = PDFResourceManager()
		# Create a PDF device object.
		device = PDFDevice(rsrcmgr)
		# Create a PDF interpreter object.
		interpreter = PDFPageInterpreter(rsrcmgr, device)

		if True:
			# Create a PDF parser object associated with the file object.
			parser = PDFParser(fp)
			try:
				# Create a PDF document object that stores the document structure.
				document = PDFDocument(parser, password=b'')
			except PDFEncryptionError as ex:
				print('PDFEncryptionError raised: {}.'.format(ex))
			except PDFSyntaxError as ex:
				print('PDFSyntaxError raised: {}.'.format(ex))
			except PDFException as ex:
				print('PDFException raised: {}.'.format(ex))
			# Check if the document allows text extraction. If not, abort.
			if not document.is_extractable:
				raise PDFTextExtractionNotAllowed

			# Page count.
			try:
				pages = resolve1(document.catalog['Pages'])
				#pages = resolve_all(document.catalog['Pages'])
				print('#pages = {}.'.format(pages['Count']))
			except KeyError as ex:
				print('KeyError raised: {}.'.format(ex))

			# Process each page contained in the document.
			for page in PDFPage.create_pages(document):
				interpreter.process_page(page)
				print('Page ID {} processed.'.format(page.pageid))
		else:
			for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=b''):  # pagenos uses zero-based indices. pagenos is sorted inside the function.
				interpreter.process_page(page)
				print('Page ID {} processed.'.format(page.pageid))
	finally:
		fp.close()

def text_extraction_example():
	import io

	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create resource manager.
		rsrcmgr = PDFResourceManager()
		# Set parameters for analysis.
		laparams = LAParams()
		retstr = io.StringIO()
		device = TextConverter(rsrcmgr, retstr, pageno=1, laparams=laparams, showpageno=False, imagewriter=None)
		interpreter = PDFPageInterpreter(rsrcmgr, device)

		for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=b''):  # pagenos uses zero-based indices. pagenos is sorted inside the function.
			interpreter.process_page(page)

			texts = retstr.getvalue()  # All texts in a page.

			print('------------------------------')
			print(texts)
	finally:
		fp.close()

def traverse_layout_object(elements):
	from collections.abc import Iterable
	from pdfminer.layout import LTChar, LTAnno, LTText, LTTextLine, LTTextLineHorizontal, LTTextLineVertical, LTTextBox, LTTextBoxHorizontal, LTTextBoxVertical, LTTextGroup
	from pdfminer.layout import LTImage, LTLine, LTCurve, LTFigure, LTRect, LTPage

	# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
	for elem in elements:
		# Preorder.
		#if isinstance(elem, Iterable):
		#	traverse_layout_object(elem)

		if isinstance(elem, LTChar):
			print('LTChar: bbox = {}, char = {}.'.format(elem.bbox, elem.get_text().replace('\n', '')))
		elif isinstance(elem, LTAnno):  # No bbox.
			print('LTAnno: letter = {}.'.format(elem.get_text().replace('\n', '')))
		elif isinstance(elem, LTText):
			print('LTText: bbox = {}, text = {}.'.format(elem.bbox, elem.get_text().replace('\n', '')))
		elif isinstance(elem, LTTextLine) or isinstance(elem, LTTextLineHorizontal) or isinstance(elem, LTTextLineVertical):
			print('LTTextLine: bbox = {}, text = {}.'.format(elem.bbox, elem.get_text().replace('\n', '')))
			#traverse_layout_object(elem)
		elif isinstance(elem, LTTextBox) or isinstance(elem, LTTextBoxHorizontal) or isinstance(elem, LTTextBoxVertical):
			print('LTTextBox: bbox = {}, text = {}.'.format(elem.bbox, elem.get_text().replace('\n', '')))
			#print('LTTextBox: coordinates = {}, size = {}.'.format((elem.x0, elem.y0, elem.x1, elem.y1), (elem.width, elem.height)))
			#traverse_layout_object(elem)
		elif isinstance(elem, LTTextGroup):
			print('LTTextGroup: bbox = {}.'.format(elem.bbox))
			#traverse_layout_object(elem)

		elif isinstance(elem, LTImage):
			#print('LTImage: bbox = {}, name = {}, mask = {}.'.format(elem.bbox, elem.name, elem.imagemask))
			print('LTImage: bbox = {}, name = {}.'.format(elem.bbox, elem.name))
		elif isinstance(elem, LTLine):
			print('LTLine: bbox = {}, points = {}.'.format(elem.bbox, elem.pts))
		elif isinstance(elem, LTCurve):
			print('LTCurve: bbox = {}, points = {}.'.format(elem.bbox, elem.pts))
		elif isinstance(elem, LTRect):
			print('LTRect: bbox = {}.'.format(elem.bbox))
		elif isinstance(elem, LTFigure):  # For PDF Form objects.
			print('LTFigure: bbox = {}, name = {}.'.format(elem.bbox, elem.name))
			#traverse_layout_object(elem)

		# Postorder.
		if isinstance(elem, Iterable):
			traverse_layout_object(elem)

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def layout_analysis_example():
	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create resource manager.
		rsrcmgr = PDFResourceManager()
		# Set parameters for analysis.
		laparams = LAParams()
		# Create a PDF page aggregator object.
		device = PDFPageAggregator(rsrcmgr, laparams=laparams)
		interpreter = PDFPageInterpreter(rsrcmgr, device)

		for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=b''):  # pagenos uses zero-based indices. pagenos is sorted inside the function.
			interpreter.process_page(page)

			# Receive the LTPage object for the page.
			layout = device.get_result()

			print('------------------------------')
			print('LTPage: bbox = {}, page ID = {}.'.format(layout.bbox, layout.pageid))  # Page ID is not a page number.
			traverse_layout_object(layout)
	finally:
		fp.close()

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def table_of_contents_example():
	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')
		# Create a PDF parser object associated with the file object.
		parser = PDFParser(fp)

		# Create a PDF document object that stores the document structure.
		document = PDFDocument(parser, password=b'')
		# Check if the document allows text extraction. If not, abort.
		if not document.is_extractable:
			raise PDFTextExtractionNotAllowed

		try:
			# Get the outlines of the document.
			outlines = document.get_outlines()
			for (level, title, dest, action, se) in outlines:
				print(level, title)
		except PDFNoOutlines as ex:
			print('No outline in {}: {}.'.format(pdf_filepath, ex))
	finally:
		fp.close()

def main():
	basic_usage()

	#text_extraction_example()
	#layout_analysis_example()
	#table_of_contents_example()

#--------------------------------------------------------------------

# The coordinate system in PDF.
#	Origin: (left, bottom).
#	X-axis: rightward.
#	Y-axis: upward.

if '__main__' == __name__:
	main()
