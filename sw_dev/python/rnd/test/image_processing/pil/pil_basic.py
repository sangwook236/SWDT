#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, glob
from PIL import Image, ImageDraw, ImageFont

# REF [site] >> http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
def image_example():
	image_filename = 'D:/dataset/pattern_recognition/street1.png'
	#image_filename = 'D:/dataset/pattern_recognition/street2.jpg'

	try:
		img = Image.open(image_filename)
	except IOError as ex:
		print('Failed to load an image:', image_filename)
		return

	img.rotate(45).show()

	#img.show()
	img.save('./pil_image.jpg')

def draw_example():
	img = Image.new(mode='RGB', size=(500, 500), color=(255, 255, 255))
	draw = ImageDraw.Draw(img)

	draw.line((50, 50, 400, 400), fill=(255, 0, 255, 0), width=2)
	draw.rectangle((100, 100, 300, 300), fill=(0, 0, 255, 0), outline=(255, 0, 0, 255))

	#img.show()
	img.save('./pil_draw.png')

def text_example():
	img = Image.new(mode='RGB', size=(100, 30), color=(128, 128, 128))
	draw = ImageDraw.Draw(img)

	text = 'Hello World'

	offset = (10, 10)
	text_size = draw.textsize(text)
	text_area = (offset[0], offset[1], offset[0] + text_size[0], offset[1] + text_size[1])

	draw.text(xy=offset, text=text, fill=(0, 0, 0))

	# Draws rectangle surrounding text.
	draw.rectangle(text_area, outline='red', width=1)

	# Crops text area.
	#img = img.crop(text_area)

	img.save('./pil_text.png')

	#--------------------
	if 'posix' == os.name:
		system_font_dir_path = '/usr/share/fonts'
		font_base_dir_path = '/home/sangwook/work/font'
	else:
		system_font_dir_path = 'C:/Windows/Fonts'
		font_base_dir_path = 'D:/work/font'
	#font_dir_path = font_base_dir_path + '/eng'
	font_dir_path = font_base_dir_path + '/kor'

	#font_filepaths = glob.glob(font_dir_path + '/*.ttf')

	if 'posix' == os.name:
		font_type = system_font_dir_path + '/truetype/FreeSans.ttf'
	else:
		font_type = system_font_dir_path + '/Arial.ttf'
		#font_type = system_font_dir_path + '/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = system_font_dir_path + '/batang.ttc'  # 바탕, 바탕체, 궁서, 궁서체.

	font = ImageFont.truetype(font=font_type, size=50, index=0)

	img = Image.new(mode='RGB', size=(200, 100), color=(128, 128, 128))
	draw = ImageDraw.Draw(img)

	text = '안녕'

	offset = (10, 10)
	text_size = draw.textsize(text, font=font)
	font_offset = font.getoffset(text)
	text_area = (offset[0], offset[1], offset[0] + text_size[0] + font_offset[0], offset[1] + text_size[1] + font_offset[1])

	draw.text(xy=offset, text=text, font=font, fill=(0, 0, 0))

	# Draws rectangle surrounding text.
	draw.rectangle(text_area, outline='red', width=1)

	# Crops text area.
	#img = img.crop(text_area)

	img.save('./pil_text_font.png')
	#img.convert('L').save('./pil_text_font.png')  # Save as a grayscale image.

def main():
	#image_example()
	draw_example()
	text_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
