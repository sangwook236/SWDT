#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/mstamy2/PyPDF2

import PyPDF2

# REF [site] >> https://www.geeksforgeeks.org/working-with-pdf-files-in-python/ 
def extract_text_example():
	filepath = './DeepLearning.pdf'
	try:
		with open(filepath, 'rb') as pdfFileObj:
			# Create a pdf reader object.
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

			# Print number of pages in pdf file.
			print('#pages = {}.'.format(pdfReader.numPages))

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
			# Create a pdf Reader object.
			pdfReader = PyPDF2.PdfFileReader(srcPdfFileObj)

			# Create a pdf writer object for new pdf.
			pdfWriter = PyPDF2.PdfFileWriter()

			# Rotate each page.
			for page in range(pdfReader.numPages):
				# Create rotated page object.
				pageObj = pdfReader.getPage(page)
				pageObj.rotateClockwise(rotation)

				# Add rotated page object to pdf writer.
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

	# Create pdf file merger object.
	pdfMerger = PyPDF2.PdfFileMerger()

	# Append pdfs one by one.
	for fpath in src_filepaths:
		try:
			with open(fpath, 'rb') as fd:
				pdfMerger.append(fd)
		except FileNotFoundError as ex:
			print('File not found, {}: {}.'.format(fpath, ex))

	try:
		# Write combined pdf to output pdf file.
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
			# Create pdf reader object.
			pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

			# Start index of first and last slices.
			start, end = 0, splits[0]

			for i in range(len(splits) + 1):
				# Create pdf writer object for (i+1)th split.
				pdfWriter = PyPDF2.PdfFileWriter()

				# Output pdf file name.
				output_filepath = filepath.split('.pdf')[0] + str(i) + '.pdf'

				# Add pages to pdf writer object.
				for page in range(start, end):
					pdfWriter.addPage(pdfReader.getPage(page))

				try:
					# Write split pdf pages to pdf file.
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
			# Create pdf reader object of watermark pdf file.
			pdfReader = PyPDF2.PdfFileReader(wmFileObj)

			# Merge watermark pdf's first page with passed page object.
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
			# Create a pdf Reader object.
			pdfReader = PyPDF2.PdfFileReader(srcPdfFileObj)

			# Create a pdf writer object for new pdf.
			pdfWriter = PyPDF2.PdfFileWriter()

			# Add watermark to each page.
			for page in range(pdfReader.numPages):
				# Create watermarked page object.
				wmpageObj = add_watermark(watermark_filepath, pdfReader.getPage(page))

				# Add watermarked page object to pdf writer.
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
	#extract_text_example()
	#rotate_page_example()
	#merge_pdf_example()  # Not correctly working.
	#split_pdf_example()
	add_watermark_example()

	# Write texts and images to PDF.
	#	REF [file] >> reportlab_test.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
