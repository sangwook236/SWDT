#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Visualize annotations.
def markup(image, annotations, font, show_bbox=True, show_text=True):
	''' Draws the segmentation, bounding box, and label of each annotation.
	'''

	# Color codes.
	bbox_color = (255, 0, 0, 255)
	txt_color = (0, 0, 255, 255)
	txt_bk_color = (255, 255, 255, 255)

	draw = ImageDraw.Draw(image, "RGBA")
	for annotation in annotations:
		bbox, txt, confidence = annotation
		bbox = [(int(xy[0]), int(xy[1])) for xy in bbox]

		if show_bbox:
			# Draw bbox.
			#draw.rectangle((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]), outline=colors[cat_name] + (255,), width=2)
			draw.polygon(bbox, outline=bbox_color, width=2)
		if show_text:
			x1, y1 = np.min(bbox, axis=0)
			x2, y2 = np.max(bbox, axis=0)

			# Draw label.
			#w, h = draw.textsize(text=txt, font=font)
			left, top, right, bottom = draw.textbbox((0, 0), text=txt, font=font)
			w, h = right - left, bottom - top
			if y2 - y1 < h:
				#draw.rectangle((bbox[0] + bbox[2], bbox[1], bbox[0] + bbox[2] + w, bbox[1] + h), fill=(64, 64, 64, 255))
				#draw.text((bbox[0] + bbox[2], bbox[1]), text=cat_name, fill=(255, 255, 255, 255), font=font)
				draw.polygon((x2, y1, x2 + w, y1 + h), fill=txt_bk_color)
				draw.text((x2, y1), text=txt, fill=txt_color, font=font)
			else:
				#draw.rectangle((bbox[0], bbox[1], bbox[0] + w, bbox[1] + h), fill=(64, 64, 64, 255))
				#draw.text((bbox[0], bbox[1]), text=cat_name, fill=(255, 255, 255, 255), font=font)
				draw.polygon((x1, y2, x1 + w, y2 + h), fill=txt_bk_color)
				draw.text((x1, y2), text=txt, fill=txt_color, font=font)
	return image

# REF [site] >> https://github.com/JaidedAI/EasyOCR
def simple_usage():
	#input_image = "./chinese.jpg"
	#input_image = "./english.png"
	#input_image = "./french.jpg"
	#input_image = "./japanese.jpg"
	#input_image = "./korean.png"
	#input_image = "./thai.jpg"
	input_image = "./dentist_screen_01.png"
	
	#reader = easyocr.Reader(["ch_sim", "en"])  # This needs to run only once to load the model into memory
	reader = easyocr.Reader(["en", "ko"], gpu=True, detect_network="craft", recog_network="standard", download_enabled=True, detector=True, recognizer=True)
	#reader = easyocr.Reader(["en", "fr", "ja", "ko"], gpu=True, model_storage_directory=None, user_network_directory=None, detect_network="craft", recog_network="standard", download_enabled=True, detector=True, recognizer=True, verbose=True, quantize=True, cudnn_benchmark=False)

	print("Recognizing...")
	start_time = time.time()
	result = reader.readtext(input_image)
	print(f"Recognized: {time.time() - start_time} secs.")
	print(f"{result}.")

	# Visualize.
	if "posix" == os.name:
		font_dir_path = "/home/sangwook/work/font"
	else:
		font_dir_path = "D:/work/font"
	#font_filepath = font_dir_path + "/DejaVuSans.ttf"
	font_filepath = font_dir_path + "/batangche.ttf"

	font = ImageFont.truetype(font_filepath, 10)
	plt.figure(figsize=(10, 10))
	with Image.open(input_image) as img:
		plt.imshow(markup(img, result, font, show_bbox=True, show_text=True))
	plt.axis("off")
	plt.tight_layout()
	plt.show()

def main():
	# Install:
	#	pip install easyocr

	simple_usage()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
