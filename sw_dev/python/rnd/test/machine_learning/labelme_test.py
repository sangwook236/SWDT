#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
sys.path.append('../../src')
sys.path.append('./src')

import os, functools, glob, json, time
import cv2

def load_data_from_json(json_filepath, flag):
	try:
		with open(json_filepath, 'r') as fd:
			json_data = json.load(fd)
	except UnicodeDecodeError as ex:
		print('[SWL] Error: Unicode decode error, {}: {}.'.format(json_filepath, ex))
		return None
	except FileNotFoundError as ex:
		print('[SWL] Error: File not found, {}: {}.'.format(json_filepath, ex))
		return None

	try:
		version = json_data['version']
		flags = json_data['flags']
		#line_color, fill_color = json_data['lineColor'], json_data['fillColor']
		try:
			line_color = json_data['lineColor']
		except KeyError as ex:
			#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
			line_color = None
		try:
			fill_color = json_data['fillColor']
		except KeyError as ex:
			#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
			fill_color = None

		dir_path = os.path.dirname(json_filepath)
		image_filepath = os.path.join(dir_path, json_data['imagePath'])
		image_data = json_data['imageData']
		image_height, image_width = json_data['imageHeight'], json_data['imageWidth']

		if True:
			img = cv2.imread(image_filepath, flag)
			if img is None:
				print('[SWL] Error: Failed to load an image, {}.'.format(image_filepath))
				return None

		shapes = list()
		for shape in json_data['shapes']:
			label, points, shape_type = shape['label'], shape['points'], shape['shape_type']
			#group_id, shape_line_color, shape_fill_color = shape['group_id'], shape['line_color'], shape['fill_color']
			try:
				group_id = shape['group_id']
			except KeyError as ex:
				#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
				group_id = None
			try:
				shape_line_color = shape['line_color']
			except KeyError as ex:
				#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
				shape_line_color = None
			try:
				shape_fill_color = shape['fill_color']
			except KeyError as ex:
				#print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
				shape_fill_color = None
			shape_dict = {
				'label': label,
				'line_color': shape_line_color,
				'fill_color': shape_fill_color,
				'points': points,
				'group_id': group_id,
				'shape_type': shape_type,
			}
			shapes.append(shape_dict)

		return {
			'version': version,
			'flags': flags,
			'shapes': shapes,
			'lineColor': line_color,
			'fillColor': fill_color,
			'imagePath': image_filepath,
			'imageData': image_data,
			'imageWidth': image_width,
			'imageHeight': image_height,
		}
	except KeyError as ex:
		print('[SWL] Warning: Key error in JSON file, {}: {}.'.format(json_filepath, ex))
		return None

def load_data_from_json_files(json_filepaths, flag):
	data_dicts = list(load_data_from_json(json_filepaths, flag) for json_filepath in json_filepaths)
	return list(dat for dat in data_dicts[0] if dat is not None)

def load_data_from_json_files_async(json_filepaths, flag):
	import multiprocessing as mp

	async_results = list()
	def async_callback(result):
		async_results.append(result)

	num_processes = 8
	#timeout = 10
	timeout = None
	#with mp.Pool(processes=num_processes, initializer=initialize_lock, initargs=(lock,)) as pool:
	with mp.Pool(processes=num_processes) as pool:
		#results = pool.map_async(functools.partial(load_data_from_json, flag=flag), json_filepaths)
		results = pool.map_async(functools.partial(load_data_from_json, flag=flag), json_filepaths, callback=async_callback)

		results.get(timeout)

	data_dicts = list(res for res in async_results[0] if res is not None)
	return data_dicts

def simple_loading_example():
	"""
	version
	flags
	shapes *
		label
		line_color
		fill_color
		points
		group_id
		shape_type
	lineColor
	fillColor
	imagePath
	imageData
	imageWidth
	imageHeight
	"""

	image_channel = 3

	if False:
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'
		data_dir_path = data_base_dir_path + '/text/layout/sminds'
		json_filepaths = sorted(glob.glob(data_dir_path + '/labelme_??/*.json', recursive=False))
	else:
		json_filepaths = sorted(glob.glob('/path/to/*.json', recursive=False))
	assert json_filepaths
	print('#JSON files = {}.'.format(len(json_filepaths)))

	#--------------------
	if 1 == image_channel:
		flag = cv2.IMREAD_GRAYSCALE
	elif 3 == image_channel:
		flag = cv2.IMREAD_COLOR
	elif 4 == image_channel:
		flag = cv2.IMREAD_ANYCOLOR  # ?
	else:
		flag = cv2.IMREAD_UNCHANGED

	print('Start creating datasets...')
	start_time = time.time()
	#data_dicts = load_data_from_json_files(json_filepaths, flag)
	data_dicts = load_data_from_json_files_async(json_filepaths, flag)
	print('End creating datasets: {} secs.'.format(time.time() - start_time))
	print('#files = {}.'.format(len(data_dicts)))

	if True:
		import random
		for idx, dat in enumerate(random.sample(data_dicts, 2)):
			print('Data #{}:'.format(idx))
			print('\tversion = {}.'.format(dat['version']))
			print('\tflags = {}.'.format(dat['flags']))
			print('\tlineColor = {}.'.format(dat['lineColor']))
			print('\tfillColor = {}.'.format(dat['fillColor']))
			print('\timagePath = {}.'.format(dat['imagePath']))
			print('\timageData = {}.'.format(dat['imageData']))
			print('\timageWidth = {}.'.format(dat['imageWidth']))
			print('\timageHeight = {}.'.format(dat['imageHeight']))

			for sidx, shape in enumerate(dat['shapes']):
				print('\tShape #{}:'.format(sidx))
				print('\t\tlabel = {}.'.format(shape['label']))
				print('\t\tline_color = {}.'.format(shape['line_color']))
				print('\t\tfill_color = {}.'.format(shape['fill_color']))
				print('\t\tpoints = {}.'.format(shape['points']))
				print('\t\tgroup_id = {}.'.format(shape['group_id']))
				print('\t\tshape_type = {}.'.format(shape['shape_type']))

	#--------------------
	num_shapes = functools.reduce(lambda nn, dat: nn + len(dat['shapes']), data_dicts, 0)
	print('#shapes = {}.'.format(num_shapes))

	shape_counts = dict()
	for dat in data_dicts:
		for shape in dat['shapes']:
			if shape['label'] in shape_counts:
				shape_counts[shape['label']] += 1
			else:
				shape_counts[shape['label']] = 1
	print('Shape labels = {}.'.format(sorted(shape_counts.keys())))
	print('#total examples = {}.'.format(sum(shape_counts.values())))
	print('#examples of each shape label = {}.'.format({k: v for k, v in sorted(shape_counts.items())}))

def main():
	simple_loading_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
