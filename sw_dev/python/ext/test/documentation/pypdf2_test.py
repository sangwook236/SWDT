#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/mstamy2/PyPDF2

import PyPDF2

# REF [site] >> https://gist.github.com/tiarno/8a2995e70cee42f01e79
def walk(obj, fnt, emb):
	'''
	If there is a key called 'BaseFont', that is a font that is used in the document.
	If there is a key called 'FontName' and another key in the same dictionary object
	that is called 'FontFilex' (where x is null, 2, or 3), then that fontname is 
	embedded.

	We create and add to two sets, fnt = fonts used and emb = fonts embedded.
	'''
	if not hasattr(obj, 'keys'):
		return None, None
	font_keys = set(['/FontFile', '/FontFile2', '/FontFile3'])
	if '/BaseFont' in obj:
		fnt.add(obj['/BaseFont'])
	if '/FontName' in obj:
		if [x for x in font_keys if x in obj]:  # Test to see if there is FontFile.
			emb.add(obj['/FontName'])

	for k in obj.keys():
		walk(obj[k], fnt, emb)

	return fnt, emb  # Return the sets for each page.

def basic_operation():
	filepath = './DeepLearning.pdf'
	try:
		with open(filepath, 'rb') as pdfFileObj:
			# Create a PDF reader object.
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

			# The number of pages in PDF file.
			print('#pages = {}.'.format(pdfReader.numPages))

			#--------------------
			# Metadata.
			doc_info = pdfReader.getDocumentInfo()  # PyPDF2.pdf.DocumentInformation.
			print('Metadata: {}.'.format(doc_info))

			print('\tAuthor = {}.'.format(doc_info.author))
			print('\tCreator = {}.'.format(doc_info.creator))
			print('\tProducer = {}.'.format(doc_info.producer))
			print('\tSubject = {}.'.format(doc_info.subject))
			print('\tTitle = {}.'.format(doc_info.title))

			print('\tAuthor (raw) = {}.'.format(doc_info.author_raw))
			print('\tCreator (raw) = {}.'.format(doc_info.creator_raw))
			print('\tProducer (raw) = {}.'.format(doc_info.producer_raw))
			print('\tSubject (raw) = {}.'.format(doc_info.subject_raw))
			print('\tTitle (raw) = {}.'.format(doc_info.title_raw))

			if '/CreationDate' in doc_info:
				print('\tCreation date = {}.'.format(doc_info['/CreationDate']))

			#--------------------
			# Font.
			# REF [site] >> https://gist.github.com/tiarno/8a2995e70cee42f01e79
			fonts, fonts_embedded = set(), set()
			for page in pdfReader.pages:
				obj = page.getObject()
				# updated via this answer:
				# https://stackoverflow.com/questions/60876103/use-pypdf2-to-detect-non-embedded-fonts-in-pdf-file-generated-by-google-docs/60895334#60895334 
				# in order to handle lists inside objects. Thanks misingnoglic !
				# untested code since I don't have such a PDF to play with.
				if type(obj) == PyPDF2.generic.ArrayObject:  # You can also do ducktyping here.
					for i in obj:
						if hasattr(i, 'keys'):
							f, e = walk(i, fonts, fonts_embedded)
							fonts = fonts.union(f)
							fonts_embedded = fonts_embedded.union(e)
				else:
					f, e = walk(obj['/Resources'], fonts, fonts_embedded)
					fonts = fonts.union(f)
					fonts_embedded = fonts_embedded.union(e)

			print('Fonts: {}.'.format(sorted(fonts)))
			font_unembedded = fonts - fonts_embedded
			if font_unembedded:
				print('Unembedded Fonts: {}.'.format(font_unembedded))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def extract_text_example():
	filepath = './DeepLearning.pdf'
	try:
		with open(filepath, 'rb') as pdfFileObj:
			# Create a PDF reader object.
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

			# Create a page object.
			pageObj = pdfReader.getPage(0)

			# Extract text from page.
			print('Text:\n{}.'.format(pageObj.extractText()))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def rotate_page_example():
	src_filepath = './DeepLearning.pdf'
	dst_filepath = './DeepLearning_rotated.pdf'
	rotation = 270

	try:
		with open(src_filepath, 'rb') as srcPdfFileObj:
			# Create a PDF Reader object.
			pdfReader = PyPDF2.PdfFileReader(srcPdfFileObj)

			# Create a PDF writer object for new PDF.
			pdfWriter = PyPDF2.PdfFileWriter()

			# Rotate each page.
			for page in range(pdfReader.numPages):
				# Create rotated page object.
				pageObj = pdfReader.getPage(page)
				pageObj.rotateClockwise(rotation)

				# Add rotated page object to PDF writer.
				pdfWriter.addPage(pageObj)

			try:
				with open(dst_filepath, 'wb') as dstPdfFileObj:
					# Write rotated pages to new file.
					pdfWriter.write(dstPdfFileObj)
			except FileNotFoundError as ex:
				print('File not found, {}: {}.'.format(dst_filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(src_filepath, ex))

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def merge_pdf_example():  
	src_filepaths = ['./DeepLearning.pdf', './DeepLearning_rotated.pdf']   
	dst_filepath  = './DeepLearning_merged.pdf'

	# Create PDF file merger object.
	pdfMerger = PyPDF2.PdfFileMerger()

	# Append pdfs one by one.
	for fpath in src_filepaths:
		try:
			with open(fpath, 'rb') as fd:
				pdfMerger.append(fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(fpath, ex))

	try:
		# Write combined PDF to output PDF file.
		with open(dst_filepath, 'wb') as fd:
			pdfMerger.write(fd)
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(dst_filepath, ex))

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def split_pdf_example():
	filepath  = './DeepLearning.pdf'
	# Split page positions.
	splits = [2, 4]
	try:
		with open(filepath, 'rb') as pdfFileObj:
			# Create PDF reader object.
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

			# Start index of first and last slices.
			start, end = 0, splits[0]

			for i in range(len(splits) + 1):
				# Create PDF writer object for (i+1)th split.
				pdfWriter = PyPDF2.PdfFileWriter()

				# Output PDF file name.
				output_filepath = filepath.split('.pdf')[0] + str(i) + '.pdf'

				# Add pages to PDF writer object.
				for page in range(start, end):
					pdfWriter.addPage(pdfReader.getPage(page))

				try:
					# Write split PDF pages to PDF file.
					with open(output_filepath, 'wb') as fd:
						pdfWriter.write(fd)
				except FileNotFoundError as ex:
					print('File not found, {}: {}.'.format(output_filepath, ex))

				# Interchange page split start position for next split.
				start = end
				try:
					# Set split end position for next split.
					end = splits[i+1]
				except IndexError:
					# Set split end position for last split.
					end = pdfReader.numPages
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

def add_watermark(watermark_filepath, pageObj):
	try:
		with open(watermark_filepath, 'rb') as wmFileObj:
			# Create PDF reader object of watermark PDF file.
			pdfReader = PyPDF2.PdfFileReader(wmFileObj)

			# Merge watermark PDF's first page with passed page object.
			pageObj.mergePage(pdfReader.getPage(0))

			# Return watermarked page object.
			return pageObj
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(watermark_filepath, ex))
		return pageObj

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def add_watermark_example():
	src_filepath = './DeepLearning.pdf'
	dst_filepath = './DeepLearning_watermarked.pdf'
	watermark_filepath = './watermark.pdf'

	try:
		with open(src_filepath, 'rb') as srcPdfFileObj:
			# Create a PDF Reader object.
			pdfReader = PyPDF2.PdfFileReader(srcPdfFileObj)

			# Create a PDF writer object for new PDF.
			pdfWriter = PyPDF2.PdfFileWriter()

			# Add watermark to each page.
			for page in range(pdfReader.numPages):
				# Create watermarked page object.
				wmpageObj = add_watermark(watermark_filepath, pdfReader.getPage(page))

				# Add watermarked page object to PDF writer.
				pdfWriter.addPage(wmpageObj)

			try:
				with open(dst_filepath, 'wb') as dstPdfFileObj:
					# Write watermarked pages to new file.
					pdfWriter.write(dstPdfFileObj)
			except FileNotFoundError as ex:
				print('File not found, {}: {}.'.format(dst_filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(src_filepath, ex))

def main():
	basic_operation()

	#extract_text_example()
	#rotate_page_example()
	#merge_pdf_example()  # Not correctly working.
	#split_pdf_example()
	#add_watermark_example()

	# Write texts and images to PDF.
	#	REF [file] >> reportlab_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
