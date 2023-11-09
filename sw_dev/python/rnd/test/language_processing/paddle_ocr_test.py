#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md
def quickstart_image():
	img_path = "./sample.png"

	# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
	# You can set the parameter 'lang' as 'ch', 'en', 'fr', 'german', 'korean', 'japan' to switch the language model in order.
	ocr = PaddleOCR(use_angle_cls=True, lang="korean")  # Need to run only once to download and load model into memory.

	print("Recognizing...")
	start_time = time.time()
	result = ocr.ocr(img_path, cls=True)
	print(f"Recognized: {time.time() - start_time} secs.")

	for idx in range(len(result)):
		res = result[idx]
		for line in res:
			print(line)

	# Draw result
	if "posix" == os.name:
		font_dir_path = "/home/sangwook/work/font"
	else:
		font_dir_path = "D:/work/font"
	#font_filepath = font_dir_path + "/DejaVuSans.ttf"
	font_filepath = font_dir_path + "/batangche.ttf"
	#font_filepath = font_dir_path + "/simfang.ttf"

	result = result[0]
	image = Image.open(img_path).convert("RGB")
	boxes = [line[0] for line in result]
	txts = [line[1][0] for line in result]
	scores = [line[1][1] for line in result]
	im_show = draw_ocr(image, boxes, txts, scores, font_path=font_filepath)
	im_show = Image.fromarray(im_show)
	im_show.save("./result.jpg")

	if False:
		plt.imshow(im_show)
		plt.axis("off")
		plt.tight_layout()
		plt.show()

# REF [site] >> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md
def quickstart_pdf():
	img_path = "./sample.pdf"

	# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
	# You can set the parameter 'lang' as 'ch', 'en', 'fr', 'german', 'korean', 'japan' to switch the language model in order.
	ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=2)  # Need to run only once to download and load model into memory

	print("Recognizing...")
	start_time = time.time()
	result = ocr.ocr(img_path, cls=True)
	print(f"Recognized: {time.time() - start_time} secs.")

	for idx in range(len(result)):
		res = result[idx]
		for line in res:
			print(line)

	# Draw result
	import numpy as np
	import cv2
	import fitz

	if "posix" == os.name:
		font_dir_path = "/home/sangwook/work/font"
	else:
		font_dir_path = "D:/work/font"
	#font_filepath = font_dir_path + "/DejaVuSans.ttf"
	font_filepath = font_dir_path + "/batangche.ttf"
	#font_filepath = font_dir_path + "/simfang.ttf"

	imgs = []
	with fitz.open(img_path) as pdf:
		for pg in range(0, pdf.pageCount):
			page = pdf[pg]
			mat = fitz.Matrix(2, 2)
			pm = page.getPixmap(matrix=mat, alpha=False)
			# If width or height > 2000 pixels, don't enlarge the image
			if pm.width > 2000 or pm.height > 2000:
				pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

			img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
			img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
			imgs.append(img)
	for idx in range(len(result)):
		res = result[idx]
		image = imgs[idx]
		boxes = [line[0] for line in res]
		txts = [line[1][0] for line in res]
		scores = [line[1][1] for line in res]
		im_show = draw_ocr(image, boxes, txts, scores, font_path=font_filepath)
		im_show = Image.fromarray(im_show)
		im_show.save(f"./result_page_{idx}.jpg")

def main():
	quickstart_image()
	#quickstart_pdf()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
