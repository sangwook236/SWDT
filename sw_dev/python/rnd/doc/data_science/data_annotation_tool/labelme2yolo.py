#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import numpy as np
from PIL import Image
import os, re

# REF [site] >> https://github.com/wkentaro/labelme
def labelme_to_yolo(json_filepath, label_list):
	try:
		with open(json_filepath, 'r') as fd:
		#with open(json_filepath, 'r', encoding='UTF8') as fd:
			json_data = json.load(fd)
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(json_filepath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(json_filepath))

		img_filepath = os.path.join(os.path.dirname(json_filepath), json_data['imagePath'])
		im = Image.open(img_filepath)
		img_width, img_height = im.size

		results = []
		shapes = json_data['shapes']
		for shape in shapes:
			try:
				object_class = label_list.index(shape['label'])
			except ValueError as ex:
				object_class = -1

			points = np.array(shape['points'])
			xy_min = np.amin(points, axis=0)
			xy_max = np.amax(points, axis=0)

			xmin, ymin, xmax, ymax = xy_min[0], xy_min[1], xy_max[0], xy_max[1]

			cx = 0.5 * (xmax + xmin) / img_width
			cy = 0.5 * (ymax + ymin) / img_height
			width = (xmax - xmin) / img_width
			height = (ymax - ymin) / img_height

			results.append((object_class, cx, cy, width, height))

		return results
	except Exception as ex:
		print('Exception raised: {} - {}'.format(ex, json_filepath))
		return None

def main():
	img_dir_path = './labelme_images'
	file_suffix = ''
	file_extension = 'json'

	label_list = ['person', 'car', 'sign']
	#label_list = ['person', 'car']

	for root, dirnames, filenames in os.walk(img_dir_path):
		filenames.sort()
		for filename in filenames:
			if re.search(file_suffix + '\.' + file_extension + '$', filename):
				filepath = os.path.join(root, filename)
				yolo_results = labelme_to_yolo(filepath, label_list)

				if yolo_results is not None:
					idx = filename.rfind('.')
					txt_filepath = os.path.join(root, filename[:idx] + '.txt')
					with open(txt_filepath, 'w') as fd:
						for result in yolo_results:
							fd.write(' '.join(map(lambda x: str(x), result)) + '\n')
		#break  # Do not include subdirectories.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
