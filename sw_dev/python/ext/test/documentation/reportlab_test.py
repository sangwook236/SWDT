#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://www.reportlab.com/

import io
import reportlab.pdfgen.canvas
import reportlab.lib.pagesizes
import reportlab.lib.utils
import PyPDF2

def write_text_and_image_example():
	src_filepath = './DeepLearning.pdf'
	dst_filepath = './DeepLearning_added.pdf'
	log_url = 'https://www.google.com/images/srpr/logo11w.png'
	img_filepath = './logo11w.png'

	# Create a new PDF with Reportlab.
	packet = io.BytesIO()
	canvas = reportlab.pdfgen.canvas.Canvas(packet, pagesize=reportlab.lib.pagesizes.letter)
	canvas.drawString(100, 100, 'Hello World !!!')
	if True:
		logo = reportlab.lib.utils.ImageReader(log_url)
		canvas.drawImage(logo, 100, 250, mask='auto')
	else:
		mask = [0, 0, 0, 0, 0, 0]
		canvas.drawImage(img_filepath, x=100, y=250, width=400, height=200, mask=mask)
	canvas.showPage()
	canvas.save()

	# Move to the beginning of the StringIO buffer.
	packet.seek(0)
	textPdfReader = PyPDF2.PdfFileReader(packet)

	try:
		with open(src_filepath, 'rb') as srcPdfFileObj:
			# Read your existing PDF.
			pdfReader = PyPDF2.PdfFileReader(srcPdfFileObj)

			pdfWriter = PyPDF2.PdfFileWriter()

			# Add the 'watermark' (which is the new pdf) on the existing page.
			page = pdfReader.getPage(0)
			page.mergePage(textPdfReader.getPage(0))
			pdfWriter.addPage(page)

			try:
				with open(dst_filepath, 'wb') as dstPdfFileObj:
					pdfWriter.write(dstPdfFileObj)
			except FileNotFoundError as ex:
				print('File not found, {}: {}.'.format(dst_filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(src_filepath, ex))

def main():
	write_text_and_image_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
