#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/pymupdf/PyMuPDF

import fitz

# REF [site] >> https://pymupdf.readthedocs.io/en/latest/tutorial.html
def tutorial():
	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a document.
		doc = fitz.open(pdf_filepath)
		#doc = fitz.Document(pdf_filepath)
	except RuntimeError as ex:
		print('RuntimeError raised: {}.'.format(ex))
		return

	print('#pages = {}.'.format(doc.page_count))
	print('#chapters = {}.'.format(doc.chapter_count))
	print('Metadata = {}.'.format(doc.metadata))

	print('Form fonts = {}.'.format(doc.FormFonts))

	print('doc.name = {}.'.format(doc.name))
	print('doc.needs_pass = {}.'.format(doc.needs_pass))
	print('doc.outline = {}.'.format(doc.outline))
	print('doc.permissions = {}.'.format(doc.permissions))

	print('doc.is_closed = {}.'.format(doc.is_closed))
	print('doc.is_dirty = {}.'.format(doc.is_dirty))
	print('doc.is_encrypted = {}.'.format(doc.is_encrypted))
	print('doc.is_form_pdf = {}.'.format(doc.is_form_pdf))
	print('doc.is_pdf = {}.'.format(doc.is_pdf))
	print('doc.is_reflowable = {}.'.format(doc.is_reflowable))
	print('doc.is_repaired = {}.'.format(doc.is_repaired))
	print('doc.last_location = {}.'.format(doc.last_location))

	print('doc.has_annots() = {}.'.format(doc.has_annots()))
	print('doc.has_links() = {}.'.format(doc.has_links()))

	print('ToC:\n{}.'.format(doc.get_toc()))

	try:
		page_no = 0
		assert page_no < doc.page_count

		print('doc.get_page_fonts(pno={}, full=False) = {}.'.format(page_no, doc.get_page_fonts(pno=page_no, full=False)))  # A list of (xref, ext, type, basefont, name, encoding, referencer (optional)).
		print('doc.get_page_images(pno={}) = {}.'.format(page_no, doc.get_page_images(pno=page_no)))
		print('doc.get_page_xobjects(pno={}) = {}.'.format(page_no, doc.get_page_xobjects(pno=page_no)))
		print('doc.get_page_pixmap(pno={}) = {}.'.format(page_no, doc.get_page_pixmap(pno=page_no)))

		print('doc.get_page_labels() = {}.'.format(doc.get_page_labels()))
		print('doc.get_page_numbers(label="label", only_one=False) = {}.'.format(doc.get_page_numbers(label='label', only_one=False)))
		print('doc.get_sigflags() = {}.'.format(doc.get_sigflags()))
		print('doc.get_xml_metadata() = {}.'.format(doc.get_xml_metadata()))

		print('doc.get_page_text(pno={}, option="text") = {}.'.format(page_no, doc.get_page_text(pno=page_no, option='text', clip=None, flags=None)))
	except IndexError as ex:
		print('IndexError raised: {}.'.format(ex))

	try:
		# Loads page number 'page_no' of the document (0-based).
		page_no = 1
		page = doc.load_page(page_id=page_no)
		#page = doc[page_no]
	except ValueError as ex:
		print('ValueError raised: {}.'.format(ex))
	except IndexError as ex:
		print('IndexError raised: {}.'.format(ex))

	for page in doc:
		print('page.number = {}.'.format(page.number))
		print('page.parent = {}.'.format(page.parent))
		print('page.rect = {}.'.format(page.rect))
		print('page.rotation = {}.'.format(page.rotation))
		print('page.xref = {}.'.format(page.xref))
		print('page.first_annot = {}.'.format(page.first_annot))
		print('page.first_link = {}.'.format(page.first_link))
		print('page.first_widget = {}.'.format(page.first_widget))
		print('page.mediabox = {}.'.format(page.mediabox))
		print('page.mediabox_size = {}.'.format(page.mediabox_size))
		print('page.cropbox = {}.'.format(page.mediabox))
		print('page.cropbox_position = {}.'.format(page.cropbox_position))
		print('page.transformation_matrix = {}.'.format(page.transformation_matrix))
		print('page.rotation_matrix = {}.'.format(page.rotation_matrix))
		print('page.derotation_matrix = {}.'.format(page.derotation_matrix))

		#links = page.get_links()  # All links of a page.
		links = page.links()  # A generator over the page’s links.
		annotations = page.annots()  # A generator over the page's annotations.
		fields = page.widgets()  # A generator over the page's form fields.

		label = page.get_label()
		draw_cmds = page.get_drawings()
		fonts = page.get_fonts(full=False)
		images = page.get_images(full=False)
		for item in images:
			image_bboxes = page.get_image_bbox(item)

		# REF [function] >> pixmap_example().
		pix = page.get_pixmap()  # fitz.Pixmap.
		#image_str = page.get_svg_image(matrix=fitz.Identity, text_as_path=True)

		# REF [function] >> text_extraction_example().
		text = page.get_text(option='text', clip=None, flags=None)  # {'text', 'blocks', 'words', 'html', 'xhtml', 'xml', 'dict', 'json', 'rawdict', 'rawjson'}.
		#text = page.get_textbox(rect)
		#text_page = page.get_textpage(clip=None, flags=3)  # fitz.TextPage.

		rects = page.search_for('mupdf')  # A list of objects of fitz.Rect.
		if rects:
			print('bl = {}, br = {}, tl = {}, tr = {}.'.format(rects[0].bottom_left, rects[0].bottom_right, rects[0].top_left, rects[0].top_right))

	#--------------------
	if False:
		# Convert an XPS file to PDF.
		xps_filepath = '/path/to/infile.xps'
		xps_doc = fitz.open(xps_filepath)
		pdf_bytes = xps_doc.convert_to_pdf()

		pdf_filepath = '/path/to/outfile.pdf'
		if True:
			pdf = fitz.open('pdf', pdf_bytes)
			pdf.save(pdf_filepath)
		else:
			pdf_out = open(pdf_filepath, 'wb')
			pdf_out.tobytes(pdf_bytes)
			pdf_out.close()

		# Copy image files to PDF pages.
		#	Each page will have image dimensions.
		image_filepaths = [
			'/path/to/image1.png',
			'/path/to/image2.png',
		]
		pdf_filepath = '/path/to/outfile.pdf'

		doc = fitz.open()  # New PDF.
		for fpath in image_filepaths:
			img_doc = fitz.open(fpath)  # Open each image as a document.
			pdf_bytes = img_doc.convert_to_pdf()  # Make a 1-page PDF.
			img_pdf = fitz.open('pdf', pdf_bytes)
			doc.insert_pdf(img_pdf)  # Insert the image PDF.
		doc.save(pdf_filepath)

	#--------------------
	"""
	doc.new_page(pno=-1, width=595, height=842)
	doc.insert_page(pno, text=None, fontsize=11, width=595, height=842, fontname="helv", fontfile=None, color=None)

	doc.copy_page(pno, to=-1)
	doc.fullcopy_page(pno, to=-1)
	doc.move_page(pno, to=-1)
	doc.delete_page(pno=-1)
	doc.delete_pages(from_page=-1, to_page=-1)

	doc.select(sequence)  # The sequence of page numbers (zero-based) to be included.

	pdf_filepath = '/path/to/new.pdf'
	doc2 = fitz.open()  # New empty PDF.
	doc2.insert_pdf(doc1, to_page=9)   # First 10 pages.
	doc2.insert_pdf(doc1, from_page=len(doc1) - 10)  # Last 10 pages.
	doc2.save(pdf_filepath)
	"""

def text_extraction_example():
	pdf_filepath = '/path/to/sample.pdf'

	try:
		# Open a document.
		doc = fitz.open(pdf_filepath)
		#doc = fitz.Document(pdf_filepath)
	except RuntimeError as ex:
		print('RuntimeError raised: {}.'.format(ex))
		return

	for page in doc:
		# page.get_text(): option = {'text', 'blocks', 'words', 'html', 'xhtml', 'xml', 'dict', 'json', 'rawdict', 'rawjson'}.
		if True:
			text = page.get_text(option='text', clip=None, flags=None)
			print('-------------------------------------------------- Text.')
			print(text)

		if False:
			# A list of text lines grouped by block. (x0, y0, x1, y1, "lines in blocks", block_no, block_type).
			#	block_type is 1 for an image block, 0 for text.
			blocks = page.get_text(option='blocks', clip=None, flags=None)
			print('-------------------------------------------------- Blocks.')
			for idx, block in enumerate(blocks):
				print('-------------------------------------------------- {}-th block.'.format(idx))
				print(block)

		if False:
			# A list of single words with bbox information. (x0, y0, x1, y1, "word", block_no, line_no, word_no).
			#	Everything wrapped in spaces is treated as a "word" with this method.
			words = page.get_text(option='words', clip=None, flags=None)
			print('-------------------------------------------------- Words.')
			print(words)
			print('#words = {}.'.format(len(words)))

		if False:
			html = page.get_text(option='html', clip=None, flags=None)
			print('-------------------------------------------------- HTML.')
			print(html)

		if False:
			xhtml = page.get_text(option='xhtml', clip=None, flags=None)
			print('-------------------------------------------------- XHTML.')
			print(xhtml)

		if False:
			xml = page.get_text(option='xml', clip=None, flags=None)
			print('-------------------------------------------------- XML.')
			print(xml)

		if False:
			page_dict = page.get_text(option='dict', clip=None, flags=None)
			print('-------------------------------------------------- dict.')
			print('Image width = {}, image height = {}.'.format(page_dict['width'], page_dict['height']))
			for idx, block in enumerate(page_dict['blocks']):
				print('-------------------------------------------------- {}-th block.'.format(idx))
				print(block['number'])  # Block no.
				print(block['type'])  # Block type: 1 for an image block, 0 for text.
				print(block['bbox'])  # Bounding box.
				print(block['lines'])  # Text lines.

		if False:
			import json

			page_json = page.get_text(option='json', clip=None, flags=None)
			print('-------------------------------------------------- JSON.')
			page_dict = json.loads(page_json)
			print('Image width = {}, image height = {}.'.format(page_dict['width'], page_dict['height']))
			for idx, block in enumerate(page_dict['blocks']):
				print('-------------------------------------------------- {}-th block.'.format(idx))
				print(block['number'])  # Block no.
				print(block['type'])  # Block type: 1 for an image block, 0 for text.
				print(block['bbox'])  # Bounding box.
				print(block['lines'])  # Text lines.

		if False:
			page_dict = page.get_text(option='rawdict', clip=None, flags=None)
			print('-------------------------------------------------- Raw dict.')
			print('Image width = {}, image height = {}.'.format(page_dict['width'], page_dict['height']))
			for idx, block in enumerate(page_dict['blocks']):
				print('-------------------------------------------------- {}-th block.'.format(idx))
				print(block['number'])  # Block no.
				print(block['type'])  # Block type: 1 for an image block, 0 for text.
				print(block['bbox'])  # Bounding box.
				for lidx, line in enumerate(block['lines']):  # Text lines. {'spans', 'wmode', 'dir', 'bbox'}.
					print('\t------------------------------ {}-th line.'.format(lidx))
					print('\t', line['wmode'])
					print('\t', line['dir'])
					print('\t', line['bbox'])
					for sidx, span in enumerate(line['spans']):  # {'size', 'flags', 'font', 'color', 'ascender', 'descender', 'chars', 'origin', 'bbox'}.
						print('\t\t-------------------- {}-th span.'.format(sidx))
						print('\t\t', span['size'])
						print('\t\t', span['flags'])
						print('\t\t', span['font'])
						print('\t\t', span['color'])
						print('\t\t', span['ascender'])
						print('\t\t', span['descender'])
						print('\t\t', span['origin'])
						print('\t\t', span['bbox'])
						for cidx, ch in enumerate(span['chars']):  # {'origin', 'bbox', 'c'}.
							print('\t\t\t---------- {}-th char.'.format(cidx))
							print('\t\t\t', ch['origin'])
							print('\t\t\t', ch['bbox'])
							print('\t\t\t', ch['c'])  # Char.

		if False:
			import json

			page_json = page.get_text(option='rawjson', clip=None, flags=None)
			print('-------------------------------------------------- Raw JSON.')
			page_dict = json.loads(page_json)
			print('Image width = {}, image height = {}.'.format(page_dict['width'], page_dict['height']))
			for idx, block in enumerate(page_dict['blocks']):
				print('-------------------------------------------------- {}-th block.'.format(idx))
				print(block['number'])  # Block no.
				print(block['type'])  # Block type: 1 for an image block, 0 for text.
				print(block['bbox'])  # Bounding box.
				for lidx, line in enumerate(block['lines']):  # Text lines. {'spans', 'wmode', 'dir', 'bbox'}.
					print('\t------------------------------ {}-th line.'.format(lidx))
					print('\t', line['wmode'])
					print('\t', line['dir'])
					print('\t', line['bbox'])
					for sidx, span in enumerate(line['spans']):  # {'size', 'flags', 'font', 'color', 'ascender', 'descender', 'chars', 'origin', 'bbox'}.
						print('\t\t-------------------- {}-th span.'.format(sidx))
						print('\t\t', span['size'])
						print('\t\t', span['flags'])
						print('\t\t', span['font'])
						print('\t\t', span['color'])
						print('\t\t', span['ascender'])
						print('\t\t', span['descender'])
						print('\t\t', span['origin'])
						print('\t\t', span['bbox'])
						for cidx, ch in enumerate(span['chars']):  # {'origin', 'bbox', 'c'}.
							print('\t\t\t---------- {}-th char.'.format(cidx))
							print('\t\t\t', ch['origin'])
							print('\t\t\t', ch['bbox'])
							print('\t\t\t', ch['c'])  # Char.

def pixmap_example():
	pdf_filepath = '/path/to/infile.pdf'
	png_filepath = '/path/to/outfile.png'

	try:
		doc = fitz.open(pdf_filepath)
	except RuntimeError as ex:
		print('RuntimeError raised: {}.'.format(ex))
		return

	try:
		page = doc.load_page(page_id=0)
	except ValueError as ex:
		print('ValueError raised: {}.'.format(ex))
		return

	#--------------------
	#pix = page.get_pixmap(matrix=fitz.Identity, colorspace=fitz.csRGB, clip=None, alpha=False, annots=True)
	pix = page.get_pixmap()

	pix.writePNG(png_filepath)
	#pix.writeImage(png_filepath)

def drawing_example():
	doc = fitz.open()
	page = doc.new_page()

	shape = page.new_shape()
	#shape = fitz.utils.Shape(page)

	if True:
		shape.shape.draw_line((100, 100), (300, 300))
		"""
		shape.draw_polyline(points)
		shape.draw_rect(rect)
		shape.draw_quad(quad)
		shape.draw_circle(center, radius)
		shape.draw_oval(tetra)
		shape.draw_sector(center, point, angle, fullSector=True)
		
		shape.draw_curve(p1, p2, p3)
		shape.draw_bezier(p1, p2, p3, p4)

		shape.insert_text(point, text, fontsize=11, fontname='helv', fontfile=None, set_simple=False, encoding=TEXT_ENCODING_LATIN, color=None, lineheight=None, fill=None, render_mode=0, border_width=1, rotate=0, morph=None, stroke_opacity=1, fill_opacity=1, oc=0)
		shape.insert_textbox(rect, buffer, fontsize=11, fontname='helv', fontfile=None, set_simple=False, encoding=TEXT_ENCODING_LATIN, color=None, fill=None, render_mode=0, border_width=1, expandtabs=8, align=TEXT_ALIGN_LEFT, rotate=0, morph=None, stroke_opacity=1, fill_opacity=1, oc=0)
		"""

	if False:
		r = fitz.Rect(100, 100, 300, 200)
		shape.draw_squiggle(r.tl, r.tr, breadth=2)
		shape.draw_squiggle(r.tr, r.br, breadth=2)
		shape.draw_squiggle(r.br, r.tl, breadth=2)

	if False:
		r = fitz.Rect(100, 100, 300, 200)
		shape.draw_zigzag(r.tl, r.tr, breadth=2)
		shape.draw_zigzag(r.tr, r.br, breadth=2)
		shape.draw_zigzag(r.br, r.tl, breadth=2)

	# fitz.utils.Shape.finish(width=1, color=None, fill=None, lineCap=0, lineJoin=0, dashes=None, closePath=True, even_odd=False, morph=(fixpoint, matrix), stroke_opacity=1, fill_opacity=1, oc=0)
	shape.finish(width=1, color=(0, 0, 1), fill=(1, 1, 0))
	shape.commit()
	doc.save('./drawing.pdf')

def transformation_example():
	if False:
		#mat = fitz.Matrix()  # (0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
		mat = fitz.Matrix(fitz.Identity)
		#mat = fitz.Matrix(1, 2, 3, 4, 5, 6)

		print('mat.a = {}.'.format(mat.a))
		print('mat.b = {}.'.format(mat.b))
		print('mat.c = {}.'.format(mat.c))
		print('mat.d = {}.'.format(mat.d))
		print('mat.e = {}.'.format(mat.e))
		print('mat.f = {}.'.format(mat.f))
		print('mat.isRectilinear = {}.'.format(mat.isRectilinear))
		
		print('mat.norm() = {}.'.format(mat.norm()))
		print('mat.preRotate(theta) = {}.'.format(mat.preRotate(theta=30)))  # [deg].
		print('mat.preScale(sx, sy) = {}.'.format(mat.preScale(sx=2, sy=1)))
		print('mat.preShear(h, v) = {}.'.format(mat.preShear(h=1, v=0)))
		print('mat.preTranslate(tx, ty) = {}.'.format(mat.preTranslate(tx=50, ty=100)))
		mat1, mat2 = fitz.Matrix(1, 0, 0, 1, 1, 0), fitz.Matrix(1, 0, 0, 1, 0, 2)
		print('mat.concat(m1, m2) = {}.'.format(mat.concat(mat1, mat2)))  # Matrix multiplication, m1 * m2.
		retval = mat.invert(mat1)
		print('mat.invert(m) = {} (retval = {}).'.format(mat, 'Invertible' if retval == 0 else 'Not invertible'))

	#--------------------
	pdf_filepath = '/path/to/sample.pdf'

	try:
		doc = fitz.open(pdf_filepath)
	except RuntimeError as ex:
		print('RuntimeError raised: {}.'.format(ex))
		return

	try:
		page = doc.load_page(page_id=0)
	except ValueError as ex:
		print('ValueError raised: {}.'.format(ex))
		return

	print('page.transformation_matrix = {}.'.format(page.transformation_matrix))  # This matrix translates coordinates from the PDF space to the MuPDF space.
	print('page.rotation_matrix = {}.'.format(page.rotation_matrix))
	print('page.derotation_matrix = {}.'.format(page.derotation_matrix))

	if False:
		# Rotate a page.
		print('page.rect = {} (before).'.format(page.rect))
		page.set_rotation(90)
		print('page.rect = {} (after).'.format(page.rect))

		# Rotate a point.
		pt = fitz.Point(0, 0)
		print('pt * page.rotation_matrix = {}.'.format(pt * page.rotation_matrix))
	else:
		mat = fitz.Matrix(fitz.Identity)
		#mat.preRotate(theta=30)  # [deg].
		#mat.preScale(sx=2, sy=0.5)
		mat.preShear(h=0, v=1)
		#mat.preTranslate(tx=50, ty=100)

		pix = page.get_pixmap(matrix=mat)

		# Visualize.
		#	REF [site] >> https://pymupdf.readthedocs.io/en/latest/tutorial.html
		from PIL import Image
		import matplotlib.pyplot as plt
		
		mode = 'RGBA' if pix.alpha else 'RGB'
		img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

		plt.figure()
		plt.imshow(img)
		plt.axis('off')
		plt.tight_layout()
		plt.show()

def main():
	#tutorial()

	text_extraction_example()
	#pixmap_example()

	#drawing_example()
	#transformation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
