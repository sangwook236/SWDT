#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
from mmocr.utils.ocr import MMOCR  # v0.6.3.

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_demo_v0_6_3(config_dir_path):
	det_model = "TextSnake"  # {"DB_r18", "DB_r50", "DRRG", "FCE_IC15", "FCE_CTW_DCNv2", "MaskRCNN_CTW", "MaskRCNN_IC15", "MaskRCNN_IC17", "PANet_CTW", "PANet_IC15", "PS_CTW", "PS_IC15", "TextSnake"}.

	# Load models into memory.
	ocr = MMOCR(det=det_model, recog=None, config_dir=config_dir_path)

	# Inference.
	#results = ocr.readtext("mmocr/demo_text_det.jpg", output="mmocr/", export="mmocr/")
	results = ocr.readtext("mmocr/demo_densetext_det.jpg", output="mmocr/", export="mmocr/")
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_recognition_demo_v0_6_3(config_dir_path):
	recog_model = "CRNN_TPS"  # {"CRNN", "SAR", "NRTR_1/16-1/8", "NRTR_1/8-1/4", "RobustScanner", "SATRN", "SATRN_sm", "SEG", "CRNN_TPS"}.

	# Load models into memory.
	ocr = MMOCR(det=None, recog=recog_model, config_dir=config_dir_path)

	# Inference.
	results = ocr.readtext("mmocr/demo_text_recog.jpg", output="mmocr/", batch_mode=True, single_batch_size=10)
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_and_recognition_demo_v0_6_3(config_dir_path):
	# Load models into memory.
	ocr = MMOCR(config_dir=config_dir_path)

	# Inference.
	results = ocr.readtext("mmocr/demo_text_ocr.jpg", print_result=True, imshow=True)
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_and_recognition_and_kie_demo_v0_6_3(config_dir_path):
	det_model = "PS_CTW"
	recog_model = "SAR"
	kie_model = "SDMGR"  # {"SDMGR"}.

	# WildReceipt dataset:
	#	https://mmocr.readthedocs.io/en/latest/datasets/kie.html

	# Load models into memory.
	ocr = MMOCR(det=det_model, recog=recog_model, kie=kie_model, config_dir=config_dir_path)

	# Inference.
	results = ocr.readtext("mmocr/demo_kie.jpeg", print_result=True, imshow=True)
	print("Results = {}.".format(results))
"""

import time
import numpy as np
from mmocr.ocr import MMOCR

# REF [site] >> https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html
def text_detection_demo(config_dir_path):
	det_model = "TextSnake"  # {DB_r18, DB_r50, DBPP_r50, DRRG, FCE_IC15, FCE_CTW_DCNv2, MaskRCNN_CTW, MaskRCNN_IC15, PANet_CTW, PANet_IC15, PS_CTW, PS_IC15, TextSnake}.
	# Available models: DB_r18 (?), DB_r50 (X), DBPP_r50, DRRG (X), FCE_IC15 (too slow), FCE_CTW_DCNv2 (slow, ?), PANet_CTW, PS_CTW, PS_IC15, TextSnake.

	# Load models into memory.
	#ocr = MMOCR(det=det_model, det_config=None, det_ckpt=None)
	ocr = MMOCR(det=det_model, det_config=None, det_ckpt=None, config_dir=config_dir_path, device="cuda")

	#-----
	# Inference.

	image_filepath = "./mmocr/demo_text_det.jpg"
	#image_filepath = "./mmocr/demo_densetext_det.jpg"

	print("Detecting texts...")
	start_time = time.time()
	results = ocr.readtext(image_filepath, img_out_dir="./mmocr_out/")
	#results = ocr.readtext(image_filepath, img_out_dir="./mmocr_out/", show=False, print_result=False, pred_out_file="")
	print("Texts detected: {} secs.".format(time.time() - start_time))

	print("Results:")
	try:
		print("\tDetection polygons: shape = {}.".format(np.asarray(results[0]["det_polygons"], dtype=np.float32).shape))
	except ValueError:
		print("\tDetection polygons (len = {}): lengths = {}.".format(len(results[0]["det_polygons"]), [len(poly) for poly in results[0]["det_polygons"]]))
	print("\tDetection scores: shape = {}.".format(np.asarray(results[0]["det_scores"], dtype=np.float32).shape))
	print("\tImage: shape = {}, dtype = {}.".format(results[1][0].shape, results[1][0].dtype))

	#-----
	image_filepaths = [
		"./mmocr/demo_text_det.jpg",
		"./mmocr/demo_densetext_det.jpg",
	]

	print("Detecting texts...")
	start_time = time.time()
	results = ocr.readtext(image_filepaths, img_out_dir="mmocr_out/")
	print("Texts detected: {} secs.".format(time.time() - start_time))

	for idx in range(len(results)):
		print("Results #{}:".format(idx))
		try:
			print("\tDetection polygons: shape = {}.".format(np.asarray(results[0][idx]["det_polygons"], dtype=np.float32).shape))
		except ValueError:
			print("\tDetection polygons (len = {}): lengths = {}.".format(len(results[0][idx]["det_polygons"]), [len(poly) for poly in results[0][idx]["det_polygons"]]))
		print("\tDetection scores: shape = {}.".format(np.asarray(results[0][idx]["det_scores"], dtype=np.float32).shape))
		print("\tImage: shape = {}, dtype = {}.".format(results[1][idx].shape, results[1][idx].dtype))

# REF [site] >> https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html
def text_detection_and_recognition_demo(config_dir_path):
	det_model = "DB_r18"  # {DB_r18, DB_r50, DBPP_r50, DRRG, FCE_IC15, FCE_CTW_DCNv2, MaskRCNN_CTW, MaskRCNN_IC15, PANet_CTW, PANet_IC15, PS_CTW, PS_IC15, TextSnake}.
	recog_model = "CRNN"  # {ABINet, ABINet_Vision, ASTER, CRNN, MASTER, NRTR_1/16-1/8, NRTR_1/8-1/4, RobustScanner, SAR, SATRN, SATRN_sm}.

	# Load models into memory.
	#ocr = MMOCR(det=det_model, recog=recog_model)
	#ocr = MMOCR(det=det_model, recog=recog_model, recog_config=None, recog_ckpt=None)
	ocr = MMOCR(det=det_model, recog=recog_model, config_dir=config_dir_path, device="cuda")

	#-----
	# Inference.

	image_filepath = "./mmocr/demo_text_ocr.jpg"
	#image_filepath = "./mmocr/demo_text_det.jpg"

	print("Recognizing texts...")
	start_time = time.time()
	results = ocr.readtext(image_filepath)
	#results = ocr.readtext(image_filepath, show=True, print_result=True)
	#results = ocr.readtext(image_filepath, show=False, print_result=False, pred_out_file="")
	print("Texts recognized: {} secs.".format(time.time() - start_time))

	print("Results:")
	try:
		print("\tDetection polygons: shape = {}.".format(np.asarray(results["det_polygons"], dtype=np.float32).shape))
	except ValueError:
		print("\tDetection polygons (len = {}): lengths = {}.".format(len(results["det_polygons"]), [len(poly) for poly in results["det_polygons"]]))
	print("\tDetection scores: shape = {}.".format(np.asarray(results["det_scores"], dtype=np.float32).shape))
	print("\tRecognition texts (len = {}): {}.".format(len(results["rec_texts"]), results["rec_texts"]))
	print("\tRecognition scores: shape = {}.".format(np.asarray(results["rec_scores"], dtype=np.float32).shape))

	#-----
	image_filepaths = [
		"./mmocr/demo_text_ocr.jpg",
		"./mmocr/demo_text_det.jpg",
	]

	print("Recognizing texts...")
	start_time = time.time()
	results = ocr.readtext(image_filepaths)
	print("Texts recognized: {} secs.".format(time.time() - start_time))

	for idx in range(len(results)):
		print("Results #{}:".format(idx))
		try:
			print("\tDetection polygons: shape = {}.".format(np.asarray(results[idx]["det_polygons"], dtype=np.float32).shape))
		except ValueError:
			print("\tDetection polygons (len = {}): lengths = {}.".format(len(results[idx]["det_polygons"]), [len(poly) for poly in results[idx]["det_polygons"]]))
		print("\tDetection scores: shape = {}.".format(np.asarray(results[idx]["det_scores"], dtype=np.float32).shape))
		print("\tRecognition texts (len = {}): {}.".format(len(results[idx]["rec_texts"]), results[idx]["rec_texts"]))
		print("\tRecognition scores: shape = {}.".format(np.asarray(results[idx]["rec_scores"], dtype=np.float32).shape))

# REF [site] >> https://mmocr.readthedocs.io/en/dev-1.x/user_guides/inference.html
def text_detection_and_recognition_and_kie_demo(config_dir_path):
	det_model = "DB_r18"
	recog_model = "CRNN"
	kie_model = "SDMGR"  # {SDMGR}.

	# WildReceipt dataset:
	#	https://mmocr.readthedocs.io/en/latest/datasets/kie.html

	# Load models into memory.
	#ocr = MMOCR(kie=kie_model, kie_config=None, kie_ckpt=None)
	#ocr = MMOCR(det=det_model, recog=recog_model, kie=kie_model, kie_config=None, kie_ckpt=None)
	ocr = MMOCR(det=det_model, recog=recog_model, kie=kie_model, config_dir=config_dir_path, device="cuda")

	#-----
	# Inference.

	image_filepath = "./mmocr/demo_kie.jpeg"

	print("Performing KIE...")
	start_time = time.time()
	#results = ocr.readtext(image_filepath, show=False, print_result=False, pred_out_file="")
	results = ocr.readtext(image_filepath, show=True, print_result=True)
	print("KIE performed: {} secs.".format(time.time() - start_time))

	print("Results = {}.".format(results))

def main():
	#config_dir_path = "/path/to/mmocr/configs"
	config_dir_path = "./mmocr/configs"

	"""
	# For v0.6.3.
	text_detection_demo_v0_6_3(config_dir_path)
	#text_recognition_demo_v0_6_3(config_dir_path)
	#text_detection_and_recognition_demo_v0_6_3(config_dir_path)
	#text_detection_and_recognition_and_kie_demo_v0_6_3(config_dir_path)
	"""
	# For v1.0.0.
	text_detection_demo(config_dir_path)
	#text_detection_and_recognition_demo(config_dir_path)
	#text_detection_and_recognition_and_kie_demo(config_dir_path)

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
