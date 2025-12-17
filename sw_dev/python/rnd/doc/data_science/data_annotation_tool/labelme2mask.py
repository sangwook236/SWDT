#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, glob, json
import numpy as np
import cv2

# REF [site] >> https://github.com/wkentaro/labelme
def labelme_to_mask(json_filepath, label_list):
	with open(json_filepath, 'r') as fd:
	#with open(json_filepath, 'r', encoding='UTF8') as fd:
		json_data = json.load(fd)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(json_filepath))
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(json_filepath))

	img_filepath = os.path.join(os.path.dirname(json_filepath), json_data['imagePath'])
	img_height, img_width = json_data['imageHeight'], json_data['imageWidth']

	split_img_filepath = os.path.splitext(img_filepath)
	mask_filepath = split_img_filepath[0] + '.mask' + split_img_filepath[1]
	#mask_filepath = split_img_filepath[0] + '.mask' + '.tif'

	mask = np.zeros((img_height, img_width), dtype=np.int16)

	shapes = json_data['shapes']
	for idx, shape in enumerate(shapes):
		try:
			object_class = label_list.index(shape['label'])
		except ValueError as ex:
			print('Invalid object label: {}.'.format(shape['label']))
			continue

		points = np.round(np.array(shape['points'])).astype(np.int32)
		#cv2.polylines(mask, [points], True, (idx + 1,), cv2.FILLED, cv2.LINE_8)
		cv2.fillPoly(mask, [points], (idx + 1,), cv2.LINE_8)

	cv2.imwrite(mask_filepath, mask)

def visualize_mask(mask_filepath, scale_factor=None):
	mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
	if mask is None:
		print('Failed to load a mask file: {}.'.format(mask_filepath))
		return

	min_val, max_val = np.min(mask), np.max(mask)
	print('Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(mask.shape, mask.dtype, min_val, max_val))

	if scale_factor is not None:
		mask = cv2.resize(mask, (round(mask.shape[1] * scale_factor), round(mask.shape[0] * scale_factor)), cv2.INTER_AREA)

	mask = (mask.astype(np.float32) - min_val) / (max_val - min_val)

	cv2.imshow('Mask', mask)
	cv2.waitKey(0)

def main():
	if True:
		label_list = ['bubble']

		json_filepaths = glob.glob('./*.json')
		for filepath in json_filepaths:
			labelme_to_mask(filepath, label_list)

	if True:
		mask_filepaths = glob.glob('./*.mask.png')
		for filepath in mask_filepaths:
			visualize_mask(filepath, scale_factor=0.2)
		cv2.destroyAllWindows()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
