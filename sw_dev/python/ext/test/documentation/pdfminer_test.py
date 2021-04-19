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

# REF [site] >> traverse_layout_object().
def extract_text_object(elements, pdf_filepath, page_idx):
	from collections.abc import Iterable
	from pdfminer.layout import LTTextLine, LTTextLineHorizontal, LTTextLineVertical
	from pdfminer.layout import LTTextBox, LTTextBoxHorizontal, LTTextBoxVertical

	bbox_text_pairs = list()
	for elem in elements:
		#if isinstance(elem, LTTextLine) or isinstance(elem, LTTextLineHorizontal) or isinstance(elem, LTTextLineVertical):
		if isinstance(elem, LTTextBox) or isinstance(elem, LTTextBoxHorizontal) or isinstance(elem, LTTextBoxVertical):
			if elem.bbox[0] >= elem.bbox[2] or elem.bbox[1] >= elem.bbox[3]:
				print('[SWL] Warning: Invalid bounding box, {} at page {} in {}.'.format(elem.bbox, page_idx, pdf_filepath))
				continue
			#bbox_text_pairs.append((elem.bbox, elem.get_text().replace('\n', '')))
			bbox_text_pairs.append((elem.bbox, elem.get_text().rstrip()))

		if isinstance(elem, Iterable):
			bbox_text_pairs.extend(extract_text_object(elem, pdf_filepath, page_idx))
	return bbox_text_pairs

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def basic_usage():
	pdf_filepath = '/path/to/sample.pdf'

	fp = None
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

			# Metadata.
			print('Metadata: {}.'.format(document.info))
			for info in document.info:
				if 'CreationDate' in info:
					print('\tCreation date = {}.'.format(info['CreationDate']))

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
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
	finally:
		if fp: fp.close()

def resource_example():
	from pdfminer.pdffont import CFFFont, TrueTypeFont
	from pdfminer.pdffont import PDFFont, PDFSimpleFont, PDFType1Font, PDFTrueTypeFont, PDFType3Font, PDFCIDFont
	from pdfminer.psparser import literal_name
	from pdfminer.pdftypes import PDFObjRef
	from pdfminer.pdftypes import list_value, dict_value, stream_value
	from pdfminer.pdfcolor import PDFColorSpace
	from pdfminer.pdfcolor import PREDEFINED_COLORSPACE

	font_filepath = '/path/to/font.ttf'
	with open(font_filepath, 'rb') as fp:
		#font = CFFFont(font_filepath, fp)
		font = TrueTypeFont(font_filepath, fp)
		print('Font type = {}.'.format(font.fonttype))
		print('Font fp = {}.'.format(font.fp))
		print('Font name = {}.'.format(font.name))
		print('Font tables = {}.'.format(font.tables))

	#--------------------
	pdf_filepath = '/path/to/sample.pdf'

	fp = None
	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create a PDF resource manager object that stores shared resources.
		rsrcmgr = PDFResourceManager()

		pages = PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=b'')  # pagenos uses zero-based indices. pagenos is sorted inside the function.
		page = next(pages)
		if page:
			resources, contents = page.resources, page.contents
			if not resources:
				print('No resource.')
				return

			if contents:
				print('Contents: {}.'.format(contents))
				#for ct in contents:
				#	print(ct.resolve())

			# REF [function] >> pdfminer.pdfinterp.PDFPageInterpreter.init_resources()
			def get_colorspace(spec):
				if isinstance(spec, list):
					name = literal_name(spec[0])
				else:
					name = literal_name(spec)
				if name == 'ICCBased' and isinstance(spec, list) and 2 <= len(spec):
					return PDFColorSpace(name, stream_value(spec[1])['N'])
				elif name == 'DeviceN' and isinstance(spec, list) and 2 <= len(spec):
					return PDFColorSpace(name, len(list_value(spec[1])))
				else:
					return PREDEFINED_COLORSPACE[name]

			fontmap, xobjmap = dict(), dict()
			csmap = PREDEFINED_COLORSPACE.copy()
			for (k, v) in dict_value(resources).items():
				#if 2 <= self.debug:
				#	print >>stderr, 'Resource: %r: %r' % (k,v)
				if k == 'Font':
					for (font_id, spec) in dict_value(v).items():
						obj_id = None
						if isinstance(spec, PDFObjRef):
							obj_id = spec.objid
						spec = dict_value(spec)
						fontmap[font_id] = rsrcmgr.get_font(obj_id, spec)
				elif k == 'ColorSpace':
					for (cs_id, spec) in dict_value(v).items():
						csmap[cs_id] = get_colorspace(resolve1(spec))
				elif k == 'ProcSet':
					rsrcmgr.get_procset(list_value(v))
				elif k == 'XObject':
					for (xobj_id, xobjstrm) in dict_value(v).items():
						xobjmap[xobj_id] = xobjstrm

			#spec = ...
			#if 'FontDescriptor' in spec:
			#	print('FontDescriptor: {}.'.format(spec['FontDescriptor'].resolve()))

			font = PDFType1Font(rsrcmgr, spec)
			font = PDFTrueTypeFont(rsrcmgr, spec)
			#font = PDFType3Font(rsrcmgr, spec)
			font = PDFCIDFont(rsrcmgr, spec)

			for font_id, font in fontmap.items():
				print('------------------------------------------------------------')
				print('Descriptor: {}.'.format(font.descriptor))
				print('\tFont name: {}, Font type: {}.'.format(font.fontname, type(font).__name__))
				if hasattr(font, 'basefont'):
					print('\tBase font: {}.'.format(font.basefont))
				if hasattr(font, 'flags'):
					print('\tFlags = {}.'.format(font.flags))
				if hasattr(font, 'default_width') and hasattr(font, 'widths'):
					print('\tDefault width = {}, Widths = {}.'.format(font.default_width, font.widths))
				print('\tAscent: {}, {}.'.format(font.ascent, font.get_ascent()))
				print('\tDescent: {}, {}.'.format(font.descent, font.get_descent()))
				if hasattr(font, 'hscale') and hasattr(font, 'vscale'):
					print('\tScale: {}, {}.'.format(font.hscale, font.vscale))
				if hasattr(font, 'leading') and hasattr(font, 'italic_angle'):
					print('\tLeading = {}, Italic angle = {}.'.format(font.leading, font.italic_angle))
				print('\tBbox = {}.'.format(font.bbox))
				if hasattr(font, 'get_width') and hasattr(font, 'get_height'):
					print('\t(width, height) = ({}, {}).'.format(font.get_width(), font.get_height()))
				if hasattr(font, 'is_multibyte') and hasattr(font, 'is_vertical'):
					print('\tis_multibyte = {}, is_vertical = {}.'.format(font.is_multibyte(), font.is_vertical()))
				if hasattr(font, 'cid2unicode') and hasattr(font, 'unicode_map'):
					print('\tcid2unicode = {}, unicode_map = {}.'.format(font.cid2unicode, font.unicode_map))
				#if hasattr(font, 'char_disp'):
				#	print('\tchar_disp({}) = {}.'.format(cid, font.char_disp(cid)))
				#if hasattr(font, 'to_unichr'):
				#	print('\tto_unichr({}) = {}.'.format(cid, font.to_unichr(cid)))
				#if hasattr(font, 'char_width') and hasattr(font, 'string_width'):
				#	print('\tchar_width({}) = {}, string_width({}) = {}.'.format(cid, font.char_width(cid), s, font.string_width(s)))
			for cs_id, cs in csmap.items():
				print('CS ID: {}.'.format(cs_id))
				print('\t{}.'.format(cs))
			for xobj_id, xobj in xobjmap.items():
				print('XObj ID: {}.'.format(xobj_id))
				print('\t{}.'.format(xobj))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
	finally:
		if fp: fp.close()

def text_extraction_example():
	import io

	pdf_filepath = '/path/to/sample.pdf'

	fp = None
	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create resource manager.
		rsrcmgr = PDFResourceManager()
		# Set parameters for layout analysis.
		laparams = LAParams(
			line_overlap=0.5,  # If two characters have more overlap than this they are considered to be on the same line.
			char_margin=2.0,  # If two characters are closer together than this margin they are considered part of the same line.
			word_margin=0.1,  # If two characters on the same line are further apart than this margin then they are considered to be two separate words, and an intermediate space will be added for readability.
			line_margin=0.5,  # If two lines are close together they are considered to be part of the same paragraph.
			boxes_flow=0.5,  # Specifies how much a horizontal and vertical position of a text matters when determining the order of text boxes.
			detect_vertical=False,  # If vertical text should be considered during layout analysis.
			all_texts=False  # If layout analysis should be performed on text in figures.
		)

		if True:
			retstr = io.StringIO()
			device = TextConverter(rsrcmgr, retstr, pageno=1, laparams=laparams, showpageno=False, imagewriter=None)
			interpreter = PDFPageInterpreter(rsrcmgr, device)

			for page in PDFPage.get_pages(fp, pagenos=None, maxpages=0, password=b''):  # pagenos uses zero-based indices. pagenos is sorted inside the function.
				interpreter.process_page(page)

				texts = retstr.getvalue()  # All texts in a page.

				print('------------------------------')
				print(texts)
		else:
			device = PDFPageAggregator(rsrcmgr, laparams=laparams)
			interpreter = PDFPageInterpreter(rsrcmgr, device)

			page_no = 1
			pages = PDFPage.get_pages(fp, pagenos=[page_no], maxpages=0, password=b'')  # pagenos uses zero-based indices. pagenos is sorted inside the function.
			page = next(pages)

			interpreter.process_page(page)

			layout = device.get_result()
			bbox_text_pairs = extract_text_object(layout, pdf_filepath, page_no)

			for idx, (bbox, txt) in enumerate(bbox_text_pairs):
				print('------------------------------ Block {} in page {} in {}.'.format(page_no, pdf_filepath))
				print(txt)
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
	finally:
		if fp: fp.close()

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def layout_analysis_example():
	pdf_filepath = '/path/to/sample.pdf'

	fp = None
	try:
		# Open a PDF file.
		fp = open(pdf_filepath, 'rb')

		# Create resource manager.
		rsrcmgr = PDFResourceManager()
		# Set parameters for layout analysis.
		laparams = LAParams(
			line_overlap=0.5,  # If two characters have more overlap than this they are considered to be on the same line.
			char_margin=2.0,  # If two characters are closer together than this margin they are considered part of the same line.
			word_margin=0.1,  # If two characters on the same line are further apart than this margin then they are considered to be two separate words, and an intermediate space will be added for readability.
			line_margin=0.5,  # If two lines are close together they are considered to be part of the same paragraph.
			boxes_flow=0.5,  # Specifies how much a horizontal and vertical position of a text matters when determining the order of text boxes.
			detect_vertical=False,  # If vertical text should be considered during layout analysis.
			all_texts=False  # If layout analysis should be performed on text in figures.
		)
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
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
	finally:
		if fp: fp.close()

# REF [site] >> https://pdfminer-docs.readthedocs.io/programming.html
def table_of_contents_example():
	pdf_filepath = '/path/to/sample.pdf'

	fp = None
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
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
	finally:
		if fp: fp.close()

def main():
	basic_usage()
	#resource_example()

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