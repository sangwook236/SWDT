#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/open-mmlab/mmocr

from mmocr.utils.ocr import MMOCR

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_demo(config_dir_path):
	det_model = "PS_IC15"  # {"DB_r18", "DB_r50", "DRRG", "FCE_IC15", "FCE_CTW_DCNv2", "MaskRCNN_CTW", "MaskRCNN_IC15", "MaskRCNN_IC17", "PANet_CTW", "PANet_IC15", "PS_CTW", "PS_IC15", "TextSnake"}.

	# Load models into memory.
	ocr = MMOCR(det=det_model, recog=None, config_dir=config_dir_path)

	# Inference.
	#results = ocr.readtext("mmocr/demo_text_det.jpg", output="mmocr/det_out.jpg", export="mmocr/")
	results = ocr.readtext("mmocr/demo_densetext_det.jpg", output="mmocr/det_out.jpg", export="mmocr/")
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_recognition_demo(config_dir_path):
	recog_model = "CRNN_TPS"  # {"CRNN", "SAR", "NRTR_1/16-1/8", "NRTR_1/8-1/4", "RobustScanner", "SATRN", "SATRN_sm", "SEG", "CRNN_TPS"}.

	# Load models into memory.
	ocr = MMOCR(det=None, recog=recog_model, config_dir=config_dir_path)

	# Inference.
	results = ocr.readtext("mmocr/demo_text_recog.jpg", output="mmocr/", batch_mode=True, single_batch_size=10)
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_and_recognition_demo(config_dir_path):
	# Load models into memory.
	ocr = MMOCR(config_dir=config_dir_path)

	# Inference.
	results = ocr.readtext("mmocr/demo_text_ocr.jpg", print_result=True, imshow=True)
	print("Results = {}.".format(results))

# REF [site] >> https://mmocr.readthedocs.io/en/latest/demo.html
def text_detection_and_recognition_and_kie_demo(config_dir_path):
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

def main():
	config_dir_path = "/home/sangwook/my_repo/python/mmocr_github/configs"

	text_detection_demo(config_dir_path)
	#text_recognition_demo(config_dir_path)
	#text_detection_and_recognition_demo(config_dir_path)
	#text_detection_and_recognition_and_kie_demo(config_dir_path)

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
