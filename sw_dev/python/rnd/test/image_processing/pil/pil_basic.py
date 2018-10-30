#!/usr/bin/env python

from PIL import Image, ImageDraw, ImageFont

# REF [site] >> http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
def simple_example():
	img_filename = 'D:/dataset/pattern_recognition/street1.png'
	#img_filename = 'D:/dataset/pattern_recognition/street2.jpg'

	img = Image.open(img_filename)

	img.rotate(45).show()

	img.save('tmp.jpg')

def text_example():
	img = Image.new(mode='RGB', size=(100, 30), color=(73, 109, 137))

	d = ImageDraw.Draw(img)
	d.text(xy=(10, 10), text='Hello World', fill=(255, 255, 0))

	img.save('./pil_text.png')

	#--------------------
	font_type = 'C:/Windows/Fonts/Arial.ttf'
	#font_type = '/Library/Fonts/Arial.ttf'
	font = ImageFont.truetype(font=font_type, size=15)

	img = Image.new(mode='RGB', size=(100, 30), color=(0, 0, 255))

	d = ImageDraw.Draw(img)
	d.text(xy=(10, 10), text='Hello World', font=font, fill=(255, 255, 255))

	img.save('./pil_text_font.png')

def main():
	#simple_example()
	text_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
