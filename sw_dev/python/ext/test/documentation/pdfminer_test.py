#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/euske/pdfminer
#	https://github.com/pdfminer/pdfminer.six

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import PDFPageAggregator

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def basic_usage():
	pdf_filepath = '/path/to/sample.pdf'

	# Open a PDF file.
	fp = open(pdf_filepath, 'rb')
	# Create a PDF parser object associated with the file object.
	parser = PDFParser(fp)

	# Create a PDF document object that stores the document structure.
	document = PDFDocument(parser, password='')
	# Check if the document allows text extraction. If not, abort.
	if not document.is_extractable:
		raise PDFTextExtractionNotAllowed

	# Create a PDF resource manager object that stores shared resources.
	rsrcmgr = PDFResourceManager()
	# Create a PDF device object.
	device = PDFDevice(rsrcmgr)

	# Create a PDF interpreter object.
	interpreter = PDFPageInterpreter(rsrcmgr, device)

	# Process each page contained in the document.
	for page in PDFPage.create_pages(document):
		interpreter.process_page(page)

	fp.close()

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def layout_analysis_example():
	pdf_filepath = '/path/to/sample.pdf'

	# Open a PDF file.
	fp = open(pdf_filepath, 'rb')
	# Create a PDF parser object associated with the file object.
	parser = PDFParser(fp)

	# Create a PDF document object that stores the document structure.
	document = PDFDocument(parser, password='')
	# Check if the document allows text extraction. If not, abort.
	if not document.is_extractable:
		raise PDFTextExtractionNotAllowed

	# Create resource manager.
	rsrcmgr = PDFResourceManager()
	# Set parameters for analysis.
	laparams = LAParams()
	# Create a PDF page aggregator object.
	device = PDFPageAggregator(rsrcmgr, laparams=laparams)

	interpreter = PDFPageInterpreter(rsrcmgr, device)
	#for page in PDFPage.get_pages(document):  # AttributeError: 'PDFDocument' object has no attribute 'seek'.
	for page in PDFPage.create_pages(document):
		interpreter.process_page(page)

		# Receive the LTPage object for the page.
		layout = device.get_result()
		for element in layout:
			if isinstance(element, LTTextBoxHorizontal):
				print(element.get_text())

	fp.close()

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def table_of_contents_example():
	pdf_filepath = '/path/to/sample.pdf'

	# Open a PDF file.
	fp = open(pdf_filepath, 'rb')
	# Create a PDF parser object associated with the file object.
	parser = PDFParser(fp)

	# Create a PDF document object that stores the document structure.
	document = PDFDocument(parser, password='')
	# Check if the document allows text extraction. If not, abort.
	if not document.is_extractable:
		raise PDFTextExtractionNotAllowed

	try:
		# Get the outlines of the document.
		outlines = document.get_outlines()
		for (level, title, dest, a, se) in outlines:
			print(level, title)
	except PDFNoOutlines as ex:
		print('No outline in {}: {}.'.format(pdf_filepath, ex))

	fp.close()

def main():
	#basic_usage()
	layout_analysis_example()
	#table_of_contents_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
