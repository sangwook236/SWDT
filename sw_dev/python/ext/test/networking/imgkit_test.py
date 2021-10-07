#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/jarrekk/imgkit

import io
import PIL.Image
import imgkit
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/jarrekk/imgkit
def simple_example():
	#imgkit.from_url('http://google.com', './out.jpg')
	#imgkit.from_file('./test.html', './out.jpg')
	imgkit.from_string('Hello!', './out.jpg')
	#imgkit.from_string(html_code_str, './out.jpg')

	input_filepath, output_filepath = './file.html', './file.jpg'
	try:
		with open(input_filepath) as fd:
			imgkit.from_file(fd, output_filepath)
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(input_filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(input_filepath, ex))

	#--------------------
	# Use False instead of output path to save PDF to a variable.
	#img_bytes = imgkit.from_url('http://google.com', False)
	#img_bytes = imgkit.from_file('./test.html', False)
	img_bytes = imgkit.from_string('Hello!', False)
	#img_bytes = imgkit.from_string(html_code_str, False)

	img = PIL.Image.open(io.BytesIO(img_bytes))

	plt.figure()
	plt.imshow(img)
	plt.tight_layout()
	plt.axis('off')
	plt.show()

def main():
	# Install.
	#	conda install wkhtmltopdf -c conda-forge
	#	pip install imgkit

	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
