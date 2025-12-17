#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, glob, json, time
import numpy as np
from PIL import Image

# REF [site] >> https://github.com/wkentaro/labelme
def labelme_to_image_patch(json_filepath, output_dir_path, image_label_info_filepath):
	try:
		try:
			with open(json_filepath, 'r') as fd:
			#with open(json_filepath, 'r', encoding='UTF8') as fd:
				json_data = json.load(fd)
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(json_filepath))
			return
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(json_filepath))
			return

		"""
		#json_data['version']
		#json_data['flags']
		json_data['shapes']
		#json_data['lineColor']
		#json_data['fillColor']
		json_data['imagePath']
		#json_data['imageData']
		json_data['imageHeight']
		json_data['imageWidth']

		json_data['shapes'][idx]['label']
		#json_data['shapes'][idx]['line_color']
		#json_data['shapes'][idx]['fill_color']
		json_data['shapes'][idx]['points']
		json_data['shapes'][idx]['shape_type']
		#json_data['shapes'][idx]['flags']
		"""

		image_filepath = os.path.join(os.path.dirname(json_filepath), json_data['imagePath'])
		img = np.array(Image.open(image_filepath))
		image_height, image_width = img.shape[:2]
		if json_data['imageHeight'] != image_height or json_data['imageWidth'] != image_width:
			print('Invalid image size: ({}, {}) != ({}, {}) in {}.'.format(image_height, image_width, json_data['imageHeight'], json_data['imageWidth'], json_data['imagePath']))
			return

		patch_dir_path = 'patch'
		os.makedirs(os.path.join(output_dir_path, patch_dir_path), exist_ok=True)

		try:
			with open(image_label_info_filepath, 'a', encoding='UTF8') as fd:
				shapes = json_data['shapes']
				for idx, shape in enumerate(shapes):
					if shape['shape_type'] in ['polygon', 'rectangle']:
						points = np.array(shape['points'])
						xy_min, xy_max = np.amin(points, axis=0), np.amax(points, axis=0)
						x1, y1, x2, y2 = round(float(xy_min[0])), round(float(xy_min[1])), round(float(xy_max[0])), round(float(xy_max[1]))
					else:
						print('Invalid shape type: {} in {}.'.format(shape['shape_type'], json_data['imagePath']))
						continue

					try:
						filename, fileext = os.path.splitext(os.path.basename(json_data['imagePath']))
						patch_filepath = os.path.join(patch_dir_path, '{0}_{2:04}{1}'.format(filename, fileext, idx))

						patch = img[y1:y2+1,x1:x2+1]
						if patch is None or 0 == patch.size:
							print('Invalid patch region: ({}, {}) - ({}, {}) in {}.'.format(x1, y1, x2, y2, json_data['imagePath']))
							continue
						Image.fromarray(patch).save(os.path.join(output_dir_path, patch_filepath))
					except KeyError as ex:
						print('File not found: {}.'.format(patch_filepath))
						continue
					except IOError as ex:
						print('Unicode decode error: {}.'.format(patch_filepath))
						continue

					fd.write('{} {}\n'.format(patch_filepath, shape['label']))
		except FileNotFoundError as ex:
			print('File not found: {}.'.format(image_label_info_filepath))
		except UnicodeDecodeError as ex:
			print('Unicode decode error: {}.'.format(image_label_info_filepath))
	except Exception as ex:
		print('Exception raised: {} - {}'.format(ex, json_filepath))

def main():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = '/work/dataset'
	data_dir_path = data_base_dir_path + '/text/receipt_epapyrus/epapyrus_20191203/image_labelme'
	output_dir_path = './labelme2patch'

	json_filepaths = sorted(glob.glob(os.path.join(data_dir_path, '*.json'), recursive=True))
	if json_filepaths is None:
		print('No JSON file found.')
		return

	os.makedirs(output_dir_path, exist_ok=False)
	image_label_info_filepath = os.path.join(output_dir_path, 'image_label_info.txt')
	print('Start generating image patches...')
	start_time = time.time()
	for json_filepath in json_filepaths:
		labelme_to_image_patch(json_filepath, output_dir_path, image_label_info_filepath)
	print('End generating image patches: {} secs.'.format(time.time() - start_time))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
