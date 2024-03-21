#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt

# REF [site] >> http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
def image_example():
	image_filename = 'D:/dataset/pattern_recognition/street1.png'
	#image_filename = 'D:/dataset/pattern_recognition/street2.jpg'

	try:
		img = Image.open(image_filename)
	except IOError as ex:
		print(f'Failed to load an image, {image_filename}: {ex}.')
		return

	print(f'{type(img)=}.')
	print(f'{isinstance(img, Image.Image)=}.')

	print(f'{img.format=}.')
	print(f'{img.size=}.')
	print(f'{img.mode=}.')

	print(f'{img.getbands()=}.')
	print(f'{img.getpixel((0, 0))=}.')
	print(f'{img.getcolors()=}.')

	#img.show()
	img.save('./pil_image.jpg')

	cropped_img = img.crop((50, 100, 150, 300))

	#cropped_img.show()
	cropped_img.save('./pil_cropped_image.jpg')

	if False:
		plt.figure()
		plt.imshow(cropped_img)
		plt.title('Image')
		plt.axis('off')
		plt.tight_layout()
		plt.show()

def draw_example():
	img = Image.new(mode='RGB', size=(500, 500), color=(255, 255, 255))
	draw = ImageDraw.Draw(img)

	draw.line((50, 50, 400, 400), fill=(255, 0, 255, 0), width=2)
	draw.rectangle((100, 100, 300, 300), fill=(0, 0, 255, 0), outline=(255, 0, 0, 255), width=1)

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

	if 'posix' == os.name:
		font_filepath = system_font_dir_path + '/truetype/FreeSans.ttf'
	else:
		font_filepath = system_font_dir_path + '/Arial.ttf'
		#font_filepath = system_font_dir_path + '/gulim.ttc'  # 굴림, 굴림체, 돋움, 돋움체.
		#font_filepath = system_font_dir_path + '/batang.ttc'  # 바탕, 바탕체, 궁서, 궁서체.

	try:
		font = ImageFont.truetype(font=font_filepath, size=50, index=0)
	except Exception as ex:
		print(f'Invalid font, {font_filepath}: {ex}.')
		raise

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

def exif_example():
	import PIL.Image
	import PIL.ExifTags

	image_filepath = '/path/to/image'
	img = PIL.Image.open(image_filepath)

	if img._getexif():
		exif_data = {PIL.ExifTags.TAGS[i]: j for i, j in img._getexif().items() if i in PIL.ExifTags.TAGS}
		print(f'EXIF: {exif_data}.')
	else:
		print('No EXIF data.')

# REF [site] >>
#	https://pillow.readthedocs.io/en/stable/reference/Image.html
#	https://realpython.com/image-processing-with-the-python-pillow-library/
def image_processing_test():
	img1 = Image.new(mode='RGBA', size=(600, 600), color=(0, 0, 0, 255))
	img2 = Image.new(mode='RGBA', size=(600, 600), color=(0, 0, 0, 255))
	img3 = Image.new(mode='RGBA', size=(600, 600), color=(0, 0, 0, 255))
	mask = Image.new(mode='1', size=(600, 600), color=0)
	img1_draw = ImageDraw.Draw(img1)
	img2_draw = ImageDraw.Draw(img2)
	img3_draw = ImageDraw.Draw(img3)
	mask_draw = ImageDraw.Draw(mask)

	img1_draw.rectangle((0, 0, 600, 100), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img1_draw.rectangle((0, 100, 600, 200), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)
	img1_draw.rectangle((0, 200, 600, 300), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img1_draw.rectangle((0, 300, 600, 400), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)
	img1_draw.rectangle((0, 400, 600, 500), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img1_draw.rectangle((0, 500, 600, 600), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)

	img2_draw.rectangle((0, 0, 100, 600), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img2_draw.rectangle((100, 0, 200, 600), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)
	img2_draw.rectangle((200, 0, 300, 600), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img2_draw.rectangle((300, 0, 400, 600), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)
	img2_draw.rectangle((400, 0, 500, 600), fill=(255, 0, 0, 255), outline=(255, 0, 0, 255), width=1)
	img2_draw.rectangle((500, 0, 600, 600), fill=(0, 0, 255, 255), outline=(0, 0, 255, 255), width=1)

	img3_draw.rectangle((0, 0, 100, 600), fill=(255, 0, 0, 63), outline=(255, 0, 0, 127), width=1)
	img3_draw.rectangle((100, 0, 200, 600), fill=(0, 0, 255, 63), outline=(0, 0, 255, 127), width=1)
	img3_draw.rectangle((200, 0, 300, 600), fill=(255, 0, 0, 63), outline=(255, 0, 0, 127), width=1)
	img3_draw.rectangle((300, 0, 400, 600), fill=(0, 0, 255, 127), outline=(0, 0, 255, 127), width=1)
	img3_draw.rectangle((400, 0, 500, 600), fill=(255, 0, 0, 127), outline=(255, 0, 0, 127), width=1)
	img3_draw.rectangle((500, 0, 600, 600), fill=(0, 0, 255, 127), outline=(0, 0, 255, 127), width=1)

	mask_draw.rectangle((0, 0, 300, 300), fill=255, outline=255, width=1)
	mask_draw.rectangle((300, 300, 600, 600), fill=255, outline=255, width=1)

	plt.figure()
	plt.imshow(img1)
	plt.title('Image 1')
	plt.axis('off')
	plt.tight_layout()

	plt.figure()
	plt.imshow(img2)
	plt.title('Image 2')
	plt.axis('off')
	plt.tight_layout()

	plt.figure()
	plt.imshow(mask)
	plt.title('Mask')
	plt.axis('off')
	plt.tight_layout()

	#-----
	result = Image.blend(img1, img2, alpha=0.5)

	plt.figure()
	plt.imshow(result)
	plt.title('blend()')
	plt.axis('off')
	plt.tight_layout()

	result = Image.composite(img1, img2, mask=mask)

	plt.figure()
	plt.imshow(result)
	plt.title('composite()')
	plt.axis('off')
	plt.tight_layout()

	result = Image.alpha_composite(img1, img3)  # Uses alpha values

	plt.figure()
	plt.imshow(result)
	plt.title('alpha_composite()')
	plt.axis('off')
	plt.tight_layout()

	rgb1 = img1.convert(mode='RGB')
	result = Image.eval(rgb1, lambda x: 255 - x)

	plt.figure()
	plt.imshow(result)
	plt.title('eval()')
	plt.axis('off')
	plt.tight_layout()

	#-----
	result = img1.copy()
	result.alpha_composite(img2, dest=(300, 300), source=(300, 300))

	plt.figure()
	plt.imshow(result)
	plt.title('alpha_composite()')
	plt.axis('off')
	plt.tight_layout()

	result = img1.copy()
	result.paste(img2, box=None, mask=mask)

	plt.figure()
	plt.imshow(result)
	plt.title('paste()')
	plt.axis('off')
	plt.tight_layout()

	#-----
	try:
		filename = './buildings.jpg'
		with Image.open(filename) as buildings_img:
			buildings_img.load()
		filename = './strawberry.jpg'
		with Image.open(filename) as strawberry_img:
			strawberry_img.load()
		filename = './cat.jpg'
		with Image.open(filename) as cat_img:
			cat_img.load()
	except FileNotFoundError as ex:
		print(f'FileNotFoundError, {filename}: {ex}.')
		return
	except IOError as ex:
		print(f'IOError, {filename}: {ex}.')
		return

	result = buildings_img.resize((buildings_img.width // 2, buildings_img.height // 2), resample=None, box=None, reducing_gap=None)

	plt.figure()
	plt.imshow(result)
	plt.title('resize()')
	plt.axis('off')
	plt.tight_layout()

	result = buildings_img.rotate(45, resample=Image.Resampling.NEAREST, expand=False, center=None, translate=None, fillcolor=None)

	plt.figure()
	plt.imshow(result)
	plt.title('rotate()')
	plt.axis('off')
	plt.tight_layout()

	result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

	plt.figure()
	plt.imshow(result)
	plt.title('transpose()')
	plt.axis('off')
	plt.tight_layout()

	#-----
	result = buildings_img.histogram(mask=None, extrema=None)

	plt.figure()
	plt.hist(result, bins=256)
	plt.title('histogram()')
	plt.axis('off')
	plt.tight_layout()

	#-----
	result = buildings_img.filter(filter=ImageFilter.BLUR)
	#result = buildings_img.filter(filter=ImageFilter.SHARPEN)
	#result = buildings_img.filter(filter=ImageFilter.SMOOTH)

	plt.figure()
	plt.imshow(result)
	plt.title('filter()')
	plt.axis('off')
	plt.tight_layout()

	#result = buildings_img.filter(filter=ImageFilter.BoxBlur(20))
	result = buildings_img.filter(filter=ImageFilter.GaussianBlur(20))

	plt.figure()
	plt.imshow(result)
	plt.title('filter()')
	plt.axis('off')
	plt.tight_layout()

	result = buildings_img.filter(filter=ImageFilter.FIND_EDGES)
	#result = buildings_img.filter(filter=ImageFilter.EDGE_ENHANCE)
	#result = buildings_img.filter(filter=ImageFilter.EMBOSS)

	plt.figure()
	plt.imshow(result)
	plt.title('filter()')
	plt.axis('off')
	plt.tight_layout()

	# Threshold
	gray = buildings_img.convert('L')
	threshold = 100
	result = gray.point(lambda x: 255 if x > threshold else 0)

	plt.figure()
	plt.imshow(result, cmap='gray')
	plt.title('point()')
	plt.axis('off')
	plt.tight_layout()

	plt.show()

def main():
	# Modes:
	#	"1": bilevel
	#	"L": greyscale
	#	"RGB": RGB
	#	"RGBA": RGBA
	#	"CMYK": CMYK
	#	"P": palette

	#image_example()
	#draw_example()
	#text_example()

	#exif_example()

	image_processing_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
