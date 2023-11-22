#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_en/quickstart_en.md
def ppocr_quickstart_image():
	import os, time
	from paddleocr import PaddleOCR, draw_ocr
	from PIL import Image
	import matplotlib.pyplot as plt

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
def ppocr_quickstart_pdf():
	import os, time
	from paddleocr import PaddleOCR, draw_ocr
	from PIL import Image

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

# REF [site] >> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/ppstructure/docs/quickstart_en.md
def ppstructure_quickstart():
	import os
	import paddleocr
	import cv2
	from PIL import Image

	paddleocr_dir_path = "/home/sangwook/my_repo/python/PaddleOCR_github"

	if True:
		# Image orientation + layout analysis + table recognition

		table_engine = paddleocr.PPStructure(show_log=True, image_orientation=True)

		save_folder = "./output_01"
		img_path = os.path.join(paddleocr_dir_path, "ppstructure/docs/table/1.png")
		img = cv2.imread(img_path)
		result = table_engine(img)
		paddleocr.save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

		for line in result:
			line.pop("img")
			print(line)

		font_path = os.path.join(paddleocr_dir_path, "doc/fonts/simfang.ttf")
		image = Image.open(img_path).convert("RGB")
		im_show = paddleocr.draw_structure_result(image, result, font_path=font_path)
		im_show = Image.fromarray(im_show)
		im_show.save("./result_01.jpg")

	if True:
		# Layout analysis + table recognition

		table_engine = paddleocr.PPStructure(show_log=True)

		save_folder = "./output_02"
		img_path = os.path.join(paddleocr_dir_path, "ppstructure/docs/table/1.png")
		img = cv2.imread(img_path)
		result = table_engine(img)
		paddleocr.save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

		for line in result:
			line.pop("img")
			print(line)

		font_path = os.path.join(paddleocr_dir_path, "doc/fonts/simfang.ttf")  # Font provided in PaddleOCR
		image = Image.open(img_path).convert("RGB")
		im_show = paddleocr.draw_structure_result(image, result, font_path=font_path)
		im_show = Image.fromarray(im_show)
		im_show.save("./result_02.jpg")

	if True:
		# Layout analysis

		table_engine = paddleocr.PPStructure(table=False, ocr=False, show_log=True)

		save_folder = "./output_03"
		img_path = os.path.join(paddleocr_dir_path, "ppstructure/docs/table/1.png")
		img = cv2.imread(img_path)
		result = table_engine(img)
		paddleocr.save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

		for line in result:
			line.pop("img")
			print(line)

	if True:
		# Table recognition

		table_engine = paddleocr.PPStructure(layout=False, show_log=True)

		save_folder = "./output_04"
		img_path = os.path.join(paddleocr_dir_path, "ppstructure/docs/table/table.jpg")
		img = cv2.imread(img_path)
		result = table_engine(img)
		paddleocr.save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

		for line in result:
			line.pop("img")
			print(line)

	if True:
		# Layout recovery

		from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx

		if True:
			# Chinese image
			table_engine = paddleocr.PPStructure(recovery=True)
		else:
			# English image
			table_engine = paddleocr.PPStructure(recovery=True, lang="en")

		save_folder = "./output_05"
		img_path = os.path.join(paddleocr_dir_path, "ppstructure/docs/table/1.png")
		img = cv2.imread(img_path)
		result = table_engine(img)
		paddleocr.save_structure_res(result, save_folder, os.path.basename(img_path).split(".")[0])

		for line in result:
			line.pop("img")
			print(line)

		h, w, _ = img.shape
		res = sorted_layout_boxes(result, w)
		convert_info_docx(img, res, save_folder, os.path.basename(img_path).split(".")[0])

def main():
	# PaddleOCR
	#ppocr_quickstart_image()
	#ppocr_quickstart_pdf()

	# PP-Structure
	ppstructure_quickstart()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
