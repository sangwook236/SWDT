#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/cocodataset/cocoapi

import time
import pycocotools.coco

# REF [site] >> https://cocodataset.org/#format-data
def simple_example():
	if True:
		annotation_filepath = '/path/to/annotation.json'
	else:
		if 'posix' == os.name:
			data_base_dir_path = '/home/sangwook/work/dataset'
		else:
			data_base_dir_path = 'D:/work/dataset'
		publaynet_dir_path = data_base_dir_path + '/text/layout/publaynet/publaynet'
		#annotation_filepath = publaynet_dir_path + '/train.json'
		annotation_filepath = publaynet_dir_path + '/val.json'

	try:
		print('Start loading a PubLayNet data from {}...'.format(publaynet_json_filepath))
		start_time = time.time()
		coco = pycocotools.coco.COCO(annotation_filepath)
		print('End loading a PubLayNet data: {} secs.'.format(time.time() - start_time))
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(annotation_filepath, ex))
		return
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(annotation_filepath, ex))
		return

	#print('Dataset: {}.'.format(coco.dataset))
	print('Dataset keys = {}.'.format(list(coco.dataset.keys())))

	if 'info' in coco.dataset:
		#print('Info: {}.'.format(coco.dataset['info']))
		print('Info: {}.'.format(coco.info()))
	if 'license' in coco.dataset:
		print('License: {}.'.format(coco.dataset['license']))

	if 'images' in coco.dataset:
		print('Images:')
		#print('\tData: {}.'.format(coco.dataset['images']))
		print('\tKeys = {}.'.format(list(coco.dataset['images'][0].keys())))
		print('\t#images = {}.'.format(len(coco.dataset['images'])))
		print('\tMin and max IDs = ({}, {}).'.format(functools.reduce(lambda mm, img: min(mm, img['id']), coco.dataset['images'], coco.dataset['images'][0]['id']), functools.reduce(lambda mm, img: max(mm, img['id']), coco.dataset['images'], 0)))
	if 'categories' in coco.dataset:
		print('Categories:')
		#print('\tData: {}.'.format(coco.dataset['categories']))
		print('\tKeys = {}.'.format(list(coco.dataset['categories'][0].keys())))
		print('\t#categories = {}.'.format(len(coco.dataset['categories'])))
		print('\tMin and max IDs = ({}, {}).'.format(functools.reduce(lambda mm, cat: min(mm, cat['id']), coco.dataset['categories'], coco.dataset['categories'][0]['id']), functools.reduce(lambda mm, cat: max(mm, cat['id']), coco.dataset['categories'], 0)))
		print('\tCategory = {}.'.format(list(cat['name'] for cat in coco.dataset['categories'])))
	if 'annotations' in coco.dataset:
		print('Annotations:')
		#print('\tData: {}.'.format(coco.dataset['annotations']))
		print('\tKeys = {}.'.format(list(coco.dataset['annotations'][0].keys())))
		print('\t#annotations = {}.'.format(len(coco.dataset['annotations'])))
		print('\tMin and max IDs = ({}, {}).'.format(functools.reduce(lambda mm, ann: min(mm, ann['id']), coco.dataset['annotations'], coco.dataset['annotations'][0]['id']), functools.reduce(lambda mm, ann: max(mm, ann['id']), coco.dataset['annotations'], 0)))
	if 'segment_infos' in coco.dataset:
		print('Segment infos:')
		#print('\tData: {}.'.format(coco.dataset['segment_infos']))
		print('\tKeys = {}.'.format(list(coco.dataset['segment_infos'][0].keys())))
		print('\t#segment infos = {}.'.format(len(coco.dataset['segment_infos'])))
		print('\tMin and max IDs = ({}, {}).'.format(functools.reduce(lambda mm, si: min(mm, si['id']), coco.dataset['segment_infos'], coco.dataset['segment_infos'][0]['id']), functools.reduce(lambda mm, si: max(mm, si['id']), coco.dataset['segment_infos'], 0)))

	#--------------------
	"""
	APIs for pycocotools.coco.COCO:
		annIds = coco.getAnnIds(imgIds=[], catIds=[], areaRng=[], iscrowd=None)
		catIds = coco.getCatIds(catNms=[], supNms=[], catIds=[])
		imgIds = coco.getImgIds(imgIds=[], catIds=[])

		anns = coco.loadAnns(ids=[])
		cats = coco.loadCats(ids=[])
		imgs = coco.loadImgs(ids=[])

		import matplotlib.pyplot as plt
		coco.showAnns(anns, draw_bbox=False)
		plt.show()

		loaded_coco = coco.loadRes(resFile)
		anns = coco.loadNumpyAnnotations(data)

		rle = coco.annToRLE(ann)
		mask = coco.annToMask(ann)
	"""

	if 'categories' in coco.dataset:
		for cat in coco.dataset['categories']:
			annIds = coco.getAnnIds(imgIds=[], catIds=[cat['id']], areaRng=[], iscrowd=None)
			print("#annotations of category '{}' = {}.".format(cat['name'], len(annIds)))

	#imgIds = coco.getImgIds(imgIds=[1, 3, 7], catIds=[1])
	#images = coco.loadImgs(ids=imgIds)
	#print('Image IDs = {}.'.format(imgIds))
	#print('Images = {}.'.format(images))

	#catIds = coco.getCatIds(catNms=[coco.dataset['categories'][1 - 1]['name'], coco.dataset['categories'][2 - 1]['name']], supNms=[], catIds=[2, 3])
	#categories = coco.loadCats(ids=annIds)
	#print('Category IDs = {}.'.format(catIds))
	#print('Categories = {}.'.format(categories))

	#annIds = coco.getAnnIds(imgIds=[1], catIds=[], areaRng=[], iscrowd=None)
	#annotations = coco.loadAnns(ids=annIds)
	#print('Annotation IDs = {}.'.format(annIds))
	#print('Annotation = {}.'.format(annotations))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
