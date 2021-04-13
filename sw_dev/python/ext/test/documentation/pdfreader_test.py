#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/maxpmaxp/pdfreader

import itertools
import pdfreader
from pdfreader import PDFDocument, SimplePDFViewer

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/tutorial.html
def document_tutorial():
	pdf_filepath = './tutorial-example.pdf'

	from io import BytesIO
	with open(pdf_filepath, 'rb') as fd:
		stream = BytesIO(fd.read())
	doc = PDFDocument(stream)

	try:
		fd = open(pdf_filepath, 'rb')
		doc = PDFDocument(fd)

		print('doc.header.version = {}.'.format(doc.header.version))
		print('doc.metadata = {}.'.format(doc.metadata))

		print('doc.root.Type = {}.'.format(doc.root.Type))
		print('doc.root.Metadata.Subtype = {}.'.format(doc.root.Metadata.Subtype))
		print('doc.root.Outlines.First["Title"] = {}.'.format(doc.root.Outlines.First['Title']))

		#--------------------
		# Browse document pages.
		page_one = next(doc.pages())

		all_pages = [p for p in doc.pages()]
		print('len(all_pages) = {}.'.format(len(all_pages)))

		page_six = next(itertools.islice(doc.pages(), 5, 6))
		page_five = next(itertools.islice(doc.pages(), 4, 5))
		page_eight = all_pages[7]

		print('page_six.MediaBox = {}.'.format(page_six.MediaBox))
		print('page_six.Annots[0].Subj = {}.'.format(page_six.Annots[0].Subj))
		print('page_six.Parent.Type = {}.'.format(page_six.Parent.Type))
		print('page_six.Parent.Count = {}.'.format(page_six.Parent.Count))
		print('len(page_six.Parent.Kids) = {}.'.format(len(page_six.Parent.Kids)))
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/tutorial.html
def content_extraction_tutorial():
	pdf_filepath = './tutorial-example.pdf'

	try:
		fd = open(pdf_filepath, 'rb')

		viewer = SimplePDFViewer(fd)
		print('viewer.metadata = {}.'.format(viewer.metadata))

		"""
		The viewer extracts:
			page images (XObject)
			page inline images (BI/ID/EI operators)
			page forms (XObject)
			decoded page strings (PDF encodings & CMap support)
			human (and robot) readable page markdown - original PDF commands containing decoded strings.
		"""

		for canvas in viewer:
			page_images = canvas.images
			page_forms = canvas.forms
			page_text = canvas.text_content  # Decoded strings with PDF markdown.
			page_inline_images = canvas.inline_images
			page_strings = canvas.strings  # Decoded plain text strings.

		viewer.navigate(8)
		viewer.render()
		page_8_canvas = viewer.canvas

		# Extract page images.
		#	There are 2 kinds of images in PDF documents:
		#		XObject images
		#		inline images

		print('len(viewer.canvas.inline_images) = {}.'.format(len(viewer.canvas.inline_images)))

		fax_image = viewer.canvas.inline_images[0]
		print('fax_image.Filter = {}.'.format(fax_image.Filter))
		print('fax_image.Width = {}, fax_image.Height = {}.'.format(fax_image.Width, fax_image.Height))

		pil_image = fax_image.to_Pillow()
		#pil_image.save('./fax-from-p8.png')

		# Extract texts.
		viewer.prev()
		print('(viewer.canvas.inline_images == []) = {}.'.format(viewer.canvas.inline_images == []))

		viewer.render()
		print('viewer.canvas.strings = {}.'.format(viewer.canvas.strings))

		viewer.navigate(1)
		viewer.render()
		print('viewer.canvas.strings = {}.'.format(viewer.canvas.strings))
		print('viewer.canvas.text_content = {}.'.format(viewer.canvas.text_content))

		#with open('./tutorial-sample-content-stream-p1.txt', 'w') as fd2:
		#	fd2.write(viewer.canvas.text_content)
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/tutorial.html
def hyperlink_and_annotation_tutorial():
	pdf_filepath = './annot-sample.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		viewer = SimplePDFViewer(fd)

		viewer.navigate(1)
		viewer.render()

		plain_text = ''.join(viewer.canvas.strings)
		print('"http" in plain_text = {}.'.format('http' in plain_text))

		print('len(viewer.annotations) = {}.'.format(len(viewer.annotations)))

		links = [annot.A.URI for annot in viewer.annotations if annot.Subtype == 'Link']
		print('links = {}.'.format(links))
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/tutorial.html
def encrypted_and_password_protected_pdf_tutorial():
	pdf_filepath = './encrypted-with-qwerty.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		viewer = SimplePDFViewer(fd, password='qwerty')

		viewer.render()

		text = ''.join(viewer.canvas.strings)
		print('text = {}.'.format(text))

		#--------------------
		doc = PDFDocument(fd, password='qwerty')

		page_one = next(doc.pages())
		print('page_one.Contents = {}.'.format(page_one.Contents))

		#--------------------
		try:
			doc = PDFDocument(fd, password='wrong password')
			#viewer = SimplePDFViewer(fd, password='wrong password')
		except ValueError as ex:
			print('ValueError raised: {}.'.format(ex))
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/extract_image.html
def xobject_image_example():
	pdf_filepath = './example-image-xobject.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		doc = PDFDocument(fd)

		# Extract XObject image.
		page = next(doc.pages())
		print('page.Resources.XObject = {}.'.format(page.Resources.XObject))

		xobj = page.Resources.XObject['img0']
		print('xobj.Type = {}, xobj.Subtype = {}.'.format(xobj.Type, xobj.Subtype))

		pil_image = xobj.to_Pillow()
		#pil_image.save('./extract-logo.png')

		#--------------------
		# Extract Images: a very simple way.
		viewer = SimplePDFViewer(fd)
		viewer.render()

		all_page_images = viewer.canvas.images
		if 'img0' in all_page_images:
			img = all_page_images['img0']
			print('img.Type = {}, img.Subtype = {}.'.format(img.Type, img.Subtype))

		all_page_inline_images = viewer.canvas.inline_images
		if all_page_inline_images:
			img = all_page_inline_images[0]
			print('img.Type = {}, img.Subtype = {}.'.format(img.Type, img.Subtype))
	finally:
		fd.close()

	#--------------------
	pdf_filepath = './tutorial-example.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		viewer = SimplePDFViewer(fd)

		# Extract image masks.
		viewer.navigate(5)
		viewer.render()

		inline_images = viewer.canvas.inline_images
		image_mask = next(img for img in inline_images if img.ImageMask)

		pil_img = image_mask.to_Pillow()
		#pil_img.save('./mask.png')
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/extract_page_text.html
def text_parsing_example():
	pdf_filepath = './example-text-crash-report.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		viewer = SimplePDFViewer(fd)
		viewer.render()

		markdown = viewer.canvas.text_content
		print('markdown = {}.'.format(markdown))

		print('viewer.canvas.strings = {}.'.format(viewer.canvas.strings))

		# Parse PDF markdown.
		print('isinstance(markdown, str) = {}.'.format(isinstance(markdown, str)))

		with open('./example-crash-markdown.txt', 'w') as fd2:
			fd2.write(markdown)

		# Now we may use any text processing tools like regular expressions, grep, custom parsers to extract the data.
		reporting_agency = markdown.split('(REPORTING AGENCY NAME *)', 1)[1].split('(', 1)[1].split(')',1)[0]
		print('reporting_agency = {}.'.format(reporting_agency))

		local_report_number = markdown.split('(LOCAL REPORT NUMBER *)', 1)[1].split('(', 1)[1].split(')',1)[0]
		print('local_report_number = {}.'.format(local_report_number))

		crash_severity = markdown.split('( ERROR)', 1)[1].split('(', 1)[1].split(')',1)[0]
		print('crash_severity = {}.'.format(crash_severity))
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/extract_form_text.html
def form_text_extraction_example():
	pdf_filepath = './example-form.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		viewer = SimplePDFViewer(fd)
		viewer.render()

		plain_text = ''.join(viewer.canvas.strings)
		print('("Farmworkers and Laborers" in plain_text) = {}.'.format('Farmworkers and Laborers' in plain_text))

		print('sorted(list(viewer.canvas.forms.keys())) = {}.'.format(sorted(list(viewer.canvas.forms.keys()))))

		form9_canvas = viewer.canvas.forms['Fm9']
		print('"".join(form9_canvas.strings) = {}.'.format(''.join(form9_canvas.strings)))
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/extract_fonts.html
def font_extraction_example():
	pdf_filepath = './example-font.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		doc = PDFDocument(fd)

		page = next(doc.pages())
		print('sorted(page.Resources.Font.keys()) = {}.'.format(sorted(page.Resources.Font.keys())))

		font = page.Resources.Font['T1_0']
		print('font.Subtype = {}, font.BaseFont = {}, font.Encoding = {}.'.format(font.Subtype, font.BaseFont, font.Encoding))

		font_file = font.FontDescriptor.FontFile
		print('type(font_file) = {}.'.format(type(font_file)))
		print('font_file.Filter = {}.'.format(font_file.Filter))

		data = font_file.filtered
		#with open('./sample-font.type1', 'wb') as fd2:
		#	 fd2.write(data)
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/extract_cmap.html
def cmap_extraction_example():
	pdf_filepath = './tutorial-example.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		doc = PDFDocument(fd)

		from itertools import islice
		page = next(islice(doc.pages(), 2, 3))
		print('page.Resources.Font = {}.'.format(page.Resources.Font))
		print('len(page.Resources.Font) = {}.'.format(len(page.Resources.Font)))

		font = page.Resources.Font['R26']
		print('font.Subtype = {}, bool(font.ToUnicode) = {}.'.format(font.Subtype, bool(font.ToUnicode)))

		# It is PostScript Type1 font, and texts use CMap provided by ToUnicode attribute.
		# Font's ToUnicode attribute contains a reference to the CMap file data stream.
		cmap = font.ToUnicode
		print('type(cmap) = {}.'.format(type(cmap)))
		print('cmap.Filter = {}.'.format(cmap.Filter))

		data = cmap.filtered
		with open('./sample-cmap.txt', 'wb') as fd2:
			 fd2.write(data)
	finally:
		fd.close()

# REF [site] >> https://pdfreader.readthedocs.io/en/latest/examples/navigate_objects.html
def pdf_object_navigation_example():
	pdf_filepath = './tutorial-example.pdf'

	try:
		fd = open(pdf_filepath, 'rb')
		doc = PDFDocument(fd)

		catalog = doc.root
		print('catalog.Type = {}.'.format(catalog.Type))
		print('catalog.Metadata.Type = {}, catalog.Metadata.Subtype = {}.'.format(catalog.Metadata.Type, catalog.Metadata.Subtype))

		pages_tree_root = catalog.Pages
		print('pages_tree_root.Type = {}.'.format(pages_tree_root.Type))

		# Attribute names are cases sensitive.
		# Missing or non-existing attributes have value of None.
		print('(catalog.type is None) = {}.'.format(catalog.type is None))
		print('(catalog.Metadata.subType is None) = {}.'.format(catalog.Metadata.subType is None))
		print('(catalog.Metadata.UnkNown_AttriBute is None) = {}.'.format(catalog.Metadata.UnkNown_AttriBute is None))

		# If object is an array, access its items by index.
		first_page = pages_tree_root.Kids[0]
		print('first_page.Type = {}.'.format(first_page.Type))
		print('first_page.Contents.Length = {}.'.format(first_page.Contents.Length))

		# If object is a stream, you can get either raw data (deflated in this example) or decoded content.
		raw_data = first_page.Contents.stream
		print('(first_page.Contents.Length == len(raw_data)) = {}.'.format(first_page.Contents.Length == len(raw_data)))
		print('first_page.Contents.Filter = {}.'.format(first_page.Contents.Filter))

		decoded_content = first_page.Contents.filtered
		print('len(decoded_content) = {}.'.format(len(decoded_content)))
		print('decoded_content.startswith(b"BT\n0 0 0 rg\n/GS0 gs") = {}.'.format(decoded_content.startswith(b'BT\n0 0 0 rg\n/GS0 gs')))

		# On the file structure level all objects have unique number and generation to identify them.
		num, gen = 2, 0
		raw_obj = doc.locate_object(num, gen)
		obj = doc.build(raw_obj)
		print('obj.Type = {}.'.format(obj.Type))
	finally:
		fd.close()

def main():
	document_tutorial()
	#content_extraction_tutorial()
	#hyperlink_and_annotation_tutorial()
	#encrypted_and_password_protected_pdf_tutorial()

	#xobject_image_example()
	#text_parsing_example()
	#form_text_extraction_example()
	#font_extraction_example()
	#cmap_extraction_example()
	#pdf_object_navigation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
