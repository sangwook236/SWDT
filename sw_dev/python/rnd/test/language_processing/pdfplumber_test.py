#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/jsvine/pdfplumber

import pdfplumber

def basic_example():
	with pdfplumber.open('./background-checks.pdf') as pdf:  # An instance of the pdfplumber.PDF class.
	#with pdfplumber.open('./background-checks.pdf', password='test') as pdf:
		print('Metadata =', pdf.metadata)

		page = pdf.pages[0]
		print('Page number =', page.page_number)  # The sequential page number.
		print('(width, height) = ({}, {})'.format(page.width, page.height))  # The page's width and height.

		if page.objects:
			print('*** Objects:', page.objects['char'][0])
			#print('*** Objects:', page.objects['line'])
			#print('*** Objects:', page.objects['rect'])
		if page.chars:
			print('*** Chars:', page.chars[0])  # A single text character.
		if page.annos:
			print('*** Annos:', page.annos[0])  # A single annotation-text character.
		if page.lines:
			print('*** Lines:', page.lines[0])  # A single 1-dimensional line.
		if page.rects:
			print('*** Rects:', page.rects[0])  # A single 2-dimensional rectangle.
		if page.curves:
			print('*** Curves:', page.curves[0])  # a series of connected points.

		"""
		page.crop(bounding_box)
		page.within_bbox(bounding_box)
		page.filter(test_function)
		page.extract_text(x_tolerance=0, y_tolerance=0)
		page.extract_words(x_tolerance=0, y_tolerance=0)
		page.extract_tables(table_settings)
		page.to_image(**conversion_kwargs)  # Returns an instance of the PageImage class.
		"""

		img = page.to_image(resolution=150)
		img.draw_rects(page.extract_words())
		img.save('./page.png', format='PNG')

# REF [file] >> ${pdfplumber_HOME}/examples/notebooks/extract-table-ca-warn-report.ipynb
def extract_table_example():
	with pdfplumber.open('./ca-warn-report.pdf') as pdf:
		page = pdf.pages[0]

		if True:
			img = page.to_image()
			img.debug_tablefinder()
			img.save('./table.png', format='PNG')

		table = page.extract_table()

		print(table[:3])

	#--------------------
	import pandas as pd

	df = pd.DataFrame(table[1:], columns=table[0])
	for column in ['Effective', 'Received']:
		df[column] = df[column].str.replace(' ', '')

	print('Table:\n', df)
		
def extract_form_value_example():
	with pdfplumber.open('path/to/form_document.pdf') as pdf:
		fields = pdf.doc.catalog['AcroForm'].resolve()['Fields']

		form_data = {}

		for field in fields:
			field_name = field.resolve()['T']
			field_value = field.resolve()['V']
			form_data[field_name] = field_value

def main():
	basic_example()

	#extract_table_example()
	#extract_form_value_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
