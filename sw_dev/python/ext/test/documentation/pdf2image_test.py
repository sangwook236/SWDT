#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Belval/pdf2image

import pdf2image
import matplotlib.pyplot as plt

def basic_example():
	pdf_filepath = '/path/to/example.pdf'

	try:
		# PIL images.
		images = pdf2image.convert_from_path(pdf_filepath, dpi=200, output_folder=None, first_page=None, last_page=None, fmt='ppm')
		#images = pdf2image.convert_from_bytes(open(pdf_filepath, 'rb').read(), dpi=200, output_folder=None, first_page=None, last_page=None, fmt='ppm')
	except pdf2image.exceptions.PDFInfoNotInstalledError as ex:
		print('PDFInfoNotInstalledError in {}: {}.'.format(pdf_filepath, ex))
		return
	except pdf2image.exceptions.PDFPageCountError as ex:
		print('PDFPageCountError in {}: {}.'.format(pdf_filepath, ex))
		return
	except pdf2image.exceptions.PDFSyntaxError as ex:
		print('PDFSyntaxError in {}: {}.'.format(pdf_filepath, ex))
		return
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(pdf_filepath, ex))
		return

	print('#images = {}.'.format(len(images)))
	for idx, img in enumerate(images):
		print('Image #{}: Size = {}, mode = {}.'.format(idx, img.size, img.mode))

		#img.save('./pdf2image_{}.png'.format(idx))
		#img.show()

		plt.imshow(img)
		plt.axis('off')
		plt.tight_layout()
		plt.show()

def main():
	basic_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
