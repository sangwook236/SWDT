#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/mittagessen/kraken

import os, functools
from kraken import binarization, pageseg, rpred
from kraken.lib import models
from kraken.lib import log
from PIL import Image

# REF [function] >> binarizer() in ${KRAKEN_HOME}/kraken/kraken.py.
def binarizer(input_image, threshold, zoom, escale, border, perc, range, low, high) -> Image:
	try:
		res = binarization.nlbin(input_image, threshold, zoom, escale, border, perc, range, low, high)
		return res
	except Exception:
		print('Binarization error.')
		raise

# REF [function] >> segmenter() in ${KRAKEN_HOME}/kraken/kraken.py.
def segmenter(input_image, text_direction, script_detect, allowed_scripts, scale, maxcolseps, black_colseps, remove_hlines, pad, mask_filepath) -> Image:
	mask = None
	if mask_filepath:
		try:
			mask = Image.open(mask_filepath)
		except IOError as e:
			print('Failed to load a mask, {}.'.format(mask_filepath))
			raise

	try:
		res = pageseg.segment(input_image, text_direction, scale, maxcolseps, black_colseps, no_hlines=remove_hlines, pad=pad, mask=mask)
		if script_detect:
			res = kraken.pageseg.detect_scripts(input_image, res, valid_scripts=allowed_scripts)
		return res
	except Exception:
		print('Page segmentation error.')
		raise

# REF [function] >> recognizer() in ${KRAKEN_HOME}/kraken/kraken.py.
def recognizer(input_image, model, pad, no_segmentation, bidi_reordering, script_ignore, mode, text_direction, segments) -> None:
	bounds = segments

	# Script detection.
	if bounds['script_detection']:
		for l in bounds['boxes']:
			for t in l:
				scripts.add(t[0])
		it = rpred.mm_rpred(model, input_image, bounds, pad,
							bidi_reordering=bidi_reordering,
							script_ignore=script_ignore)
	else:
		it = rpred.rpred(model['default'], input_image, bounds, pad,
						 bidi_reordering=bidi_reordering)

	preds = []
	with log.progressbar(it, label='Processing', length=len(bounds['boxes'])) as bar:
		for pred in bar:
			preds.append(pred)

	#--------------------
	print('Recognition results = {}.'.format('\n'.join(s.prediction for s in preds)))

	if False:
		with open_file(output, 'w', encoding='utf-8') as fp:
			print('Serializing as {} into {}'.format(mode, output))
			if mode != 'text':
				from kraken import serialization
				fp.write(serialization.serialize(preds, base_image,
												 Image.open(base_image).size,
												 text_direction,
												 scripts,
												 mode))
			else:
				fp.write('\n'.join(s.prediction for s in preds))

def simple_example():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/work/dataset'
	else:
		data_dir_path = 'D:/work/dataset'
	image_filepath = data_dir_path + '/text/receipt_epapyrus/keit_20190619/크기변환_카드영수증_5-1.png'
	#image_filepath = data_dir_path + '/text/receipt_epapyrus/epapyrus_20190618/receipt_1/img01.jpg'

	try:
		input_image = Image.open(image_filepath)
	except IOError:
		print('Failed to load an image, {}.'.format(image_filepath))
		return

	#--------------------
	threshold =0.5
	zoom = 0.5
	escale = 1.0
	border = 0.1
	perc = 80  # [1, 100].
	range = 20
	low = 5  # [1, 100].
	high = 90  # [1, 100].

	binary = binarizer(input_image, threshold, zoom, escale, border, perc, range, low, high)

	#--------------------
	text_direction = 'horizontal-lr'  # Sets principal text direction. {'horizontal-lr', 'horizontal-rl', 'vertical-lr', 'vertical-rl'}.
	script_detect = False  # Enable script detection on segmenter output.
	allowed_scripts = None  # List of allowed scripts in script detection output. Ignored if disabled.
	scale = None
	maxcolseps = 2
	black_colseps = False
	remove_hlines = True
	pad = (0, 0)  # Left and right padding around lines.
	mask_filepath = None  # Segmentation mask suppressing page areas for line detection. 0-valued image regions are ignored for segmentation purposes. Disables column detection.

	segments = segmenter(binary, text_direction, script_detect, allowed_scripts, scale, maxcolseps, black_colseps, remove_hlines, pad, mask_filepath)
	# segments.keys() = ['text_direction', 'boxes', 'script_detection'].

	#--------------------
	# Visualize bounding boxes.
	if False:
		import cv2
		rgb = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
		if rgb is None:
			print('Failed to load an image file, {}.'.format(image_filepath))
			return
		else:
			for bbox in segments['boxes']:
				x0, y0, x1, y1 = bbox
				cv2.rectangle(rgb, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
			cv2.imshow('Image', rgb)
			cv2.waitKey(0)

	#--------------------
	# Download model.
	#	kraken.py get 10.5281/zenodo.2577813
	#	python kraken.py get 10.5281/zenodo.2577813
	#		~/.config/kraken
	#		~/.kraken
	#		/usr/local/share/ocropus
	#DEFAULT_MODEL = 'en-default.mlmodel'
	DEFAULT_MODEL = './en_best.mlmodel'

	#model = DEFAULT_MODEL  # Path to an recognition model or mapping of the form $script1:$model1. Add multiple mappings to run multi-model recognition based on detected scripts. Use the default keyword for adding a catch-all model. Recognition on scripts can be ignored with the model value ignore.
	pad = 16 # Left and right padding around lines.
	reorder = True  # Reorder code points to logical order.
	no_segmentation = False  # Enables non-segmentation mode treating each input image as a whole line.
	serializer = 'text'  # Switch between hOCR, ALTO, and plain text output. {'hocr', 'alto', 'abbyyxml', 'text'}.
	text_direction = 'horizontal-tb'  # Sets principal text direction in serialization output. {'horizontal-tb', 'vertical-lr', 'vertical-rl'}.
	#lines = 'lines.json'  # JSON file containing line coordinates.
	threads = 1  # Number of threads to use for OpenMP parallelization.
	device = 'cpu'  # Select device to use (cpu, cuda:0, cuda:1, ...).

	model_dict = {'ignore': []}  # type: Dict[str, Union[str, List[str]]]
	model_dict['default'] = DEFAULT_MODEL

	nm = {}  # type: Dict[str, models.TorchSeqRecognizer].
	ign_scripts = model_dict.pop('ignore')
	for k, v in model_dict.items():
		location = None
		if os.path.isfile(v):
			location = v
		if not location:
			print('No model {} for {} found.'.format(v, k))
			continue

		try:
			rnn = models.load_any(location, device=device)
			nm[k] = rnn
		except Exception:
			print('Model loading error, {}.'.format(location))
			continue

	if 'default' in nm:
		from collections import defaultdict

		nn = defaultdict(lambda: nm['default'])  # type: Dict[str, models.TorchSeqRecognizer].
		nn.update(nm)
		nm = nn
	else:
		print('No default model.')
		return
	# Thread count is global so setting it once is sufficient.
	nn[k].nn.set_num_threads(threads)

	return recognizer(input_image, model=nm,
				   pad=pad,
				   no_segmentation=no_segmentation,
				   bidi_reordering=reorder,
				   script_ignore=ign_scripts,
				   mode=serializer,
				   text_direction=text_direction,
				   segments=segments)

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
