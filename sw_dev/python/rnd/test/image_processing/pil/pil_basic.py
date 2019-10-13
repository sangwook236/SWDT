#!/usr/bin/env python

import os
from PIL import Image, ImageDraw, ImageFont

# REF [site] >> http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
def image_example():
	img_filename = 'D:/dataset/pattern_recognition/street1.png'
	#img_filename = 'D:/dataset/pattern_recognition/street2.jpg'

	try:
		img = Image.open(img_filename)
	except IOError as ex:
		print('Failed to load an image:', img_filename)
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
		system_font_dir = '/usr/share/fonts'
		font_dir = '/home/sangwook/work/font'
	else:
		system_font_dir = 'C:/Windows/Fonts'
		font_dir = 'D:/work/font'

	if 'posix' == os.name:
		#font_type = system_font_dir + '/truetype/FreeSans.ttf'
		pass
	else:
		#font_type = system_font_dir + '/Arial.ttf'
		#font_type = system_font_dir + '/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_type = system_font_dir + '/batang.ttc'  # 바탕, 바탕체, 궁서, 궁서체.
		pass
	font_type = font_dir + '/gulim.ttf'  # 굴림, 굴림체, 돋움, 돋움체.
	#font_type = font_dir + '/batang.ttf'  # 바탕, 바탕체, 궁서, 궁서체.
	#font_type = font_dir + '/gabia_bombaram.ttf'  # (O).
	#font_type = font_dir + '/gabia_napjakBlock.ttf'  # (O).
	#font_type = font_dir + '/gabia_solmee.ttf'  # (O).
	#font_type = font_dir + '/godoMaum.ttf'  # (O).
	#font_type = font_dir + '/godoRounded L.ttf'  # (X).
	#font_type = font_dir + '/godoRounded R.ttf'  # (X).
	#font_type = font_dir + '/HS두꺼비체.ttf'
	#font_type = font_dir + '/HS봄바람체1.0.ttf'
	#font_type = font_dir + '/HS봄바람체2.0.ttf'  # (O).
	#font_type = font_dir + '/HS겨울눈꽃체.ttf'  # (O).
	#font_type = font_dir + '/HS여름물빛체.ttf'
	#font_type = font_dir + '/HS가을생각체1.0 Regular.ttf'
	#font_type = font_dir + '/HS가을생각체1.0 Thin.ttf'
	#font_type = font_dir + '/HS가을생각체2.0.ttf'  # (O).
	#font_type = font_dir + '/NanumBarunGothic.ttf'
	#font_type = font_dir + '/NanumBarunGothicBold.ttf'
	#font_type = font_dir + '/NanumBarunGothicLight.ttf'
	#font_type = font_dir + '/NanumBarunGothicUltraLight.ttf'
	#font_type = font_dir + '/NanumBarunpenB.ttf'
	#font_type = font_dir + '/NanumBarunpenR.ttf'
	#font_type = font_dir + '/NanumBrush.ttf'  # (O).
	#font_type = font_dir + '/NanumGothic.ttf'
	#font_type = font_dir + '/NanumGothicBold.ttf'
	#font_type = font_dir + '/NanumGothicExtraBold.ttf'
	#font_type = font_dir + '/NanumGothicLight.ttf'
	#font_type = font_dir + '/NanumMyeongjo.ttf'
	#font_type = font_dir + '/NanumMyeongjoBold.ttf'
	#font_type = font_dir + '/NanumMyeongjoExtraBold.ttf'
	#font_type = font_dir + '/NanumPen.ttf'  # (O).
	#font_type = font_dir + '/NanumSquareB.ttf'
	#font_type = font_dir + '/NanumSquareEB.ttf'
	#font_type = font_dir + '/NanumSquareL.ttf'
	#font_type = font_dir + '/NanumSquareR.ttf'
	#font_type = font_dir + '/NanumSquareRoundB.ttf'
	#font_type = font_dir + '/NanumSquareRoundEB.ttf'
	#font_type = font_dir + '/NanumSquareRoundL.ttf'
	#font_type = font_dir + '/NanumSquareRoundR.ttf'
	#font_type = font_dir + '/SDMiSaeng.ttf'  # (O).

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
