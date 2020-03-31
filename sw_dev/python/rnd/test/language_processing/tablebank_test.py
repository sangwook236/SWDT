#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/doc-analysis/TableBank
#	https://github.com/facebookresearch/detectron2
#	https://detectron2.readthedocs.io/

import os, random, json
import numpy as np
import torch, torchvision
import cv2
import detectron2
import detectron2.model_zoo, detectron2.config, detectron2.engine, detectron2.evaluation, detectron2.utils.visualizer, detectron2.data, detectron2.structures

def check_tablebank_detection_data(data_dir_path):
	latex_image_dir_path = os.path.join(data_dir_path, 'Detection_data/Latex/images')
	latex_json_filepath = os.path.join(data_dir_path, 'Detection_data/Latex/Latex.json')
	word_image_dir_path = os.path.join(data_dir_path, 'Detection_data/Word/images')
	word_json_filepath = os.path.join(data_dir_path, 'Detection_data/Word/Word.json')

	with open(latex_json_filepath) as fd:
		latex_annos = json.load(fd)
	print('Info:', latex_annos['info'])  # {'contributor': 'MSRA NLC Group', 'description': 'TableBank Dataset', 'version': '1.0', 'url': '', 'year': 2019, 'date_created': '2019/02/28'}.
	print('Licenses:', latex_annos['licenses'])  # [{'id': 1, 'url': 'https://creativecommons.org/licenses/by-nc-nd/4.0/', 'name': 'Attribution-NonCommercial-NoDerivs License'}].
	print('Categories:', latex_annos['categories'])  # [{'id': 1, 'supercategory': 'table', 'name': 'table'}].
	print("len(latex_annos['images']) =", len(latex_annos['images']))
	print("len(latex_annos['annotations']) =", len(latex_annos['annotations']))
	#print('Images:', latex_annos['images'])  # [{'file_name': '1401.0007_15.jpg', 'id': 1, 'license': 1, 'width': 596, 'height': 842}, ...].
	#print('Annotations:', latex_annos['annotations'])  # [{'segmentation': [[85, 396, 85, 495, 510, 495, 510, 396]], 'area': 42075, 'image_id': 1, 'category_id': 1, 'id': 1, 'iscrowd': 0, 'bbox': [85, 396, 425, 99]}, ...].

	with open(word_json_filepath) as fd:
		word_annos = json.load(fd)
	print('Info:', word_annos['info'])  # {'year': 2019, 'contributor': 'MSRA NLC Group', 'date_created': '2019/02/28', 'version': '1.0', 'url': '', 'description': 'TableBank Dataset'}.
	print('Licenses:', word_annos['licenses'])  # [{'url': 'https://creativecommons.org/licenses/by-nc-nd/4.0/', 'id': 1, 'name': 'Attribution-NonCommercial-NoDerivs License'}].
	print('Categories:', word_annos['categories'])  # [{'id': 1, 'name': 'table', 'supercategory': 'table'}].
	print("len(word_annos['images']) =", len(word_annos['images']))
	print("len(word_annos['annotations']) =", len(word_annos['annotations']))
	#print('Images:', word_annos['images'])  # [{'file_name': '%20Edward%20Dawes%20paper_13.jpg', 'width': 596, 'id': 1, 'height': 842, 'license': 1}, ...].
	#print('Annotations:', word_annos['annotations'])  #  [{'category_id': 1, 'area': 46172, 'iscrowd': 0, 'segmentation': [[89, 316, 89, 435, 477, 435, 477, 316]], 'id': 1, 'image_id': 1, 'bbox': [89, 316, 388, 119]}, ...].

	for idx, img_info in enumerate(latex_annos['images']):
		if img_info['id'] != idx + 1:
			print('Mismatch Index: {}, {} != {}.'.format(img_info['file_name'], img_info['id'], idx))
	for idx, img_info in enumerate(word_annos['images']):
		if img_info['id'] != idx + 1:
			print('Mismatch Index: {}, {} != {}.'.format(img_info['file_name'], img_info['id'], idx))

	latex_annos['annotations'] = sorted(latex_annos['annotations'], key=lambda item: item['image_id'])
	word_annos['annotations'] = sorted(word_annos['annotations'], key=lambda item: item['image_id'])

	annos_lst = [latex_annos, word_annos]
	image_dir_path_lst = [latex_image_dir_path, word_image_dir_path]
	prev_image_filename = ''
	for annos, image_dir_path in zip(annos_lst, image_dir_path_lst):
		for idx, anno_info in enumerate(annos['annotations']):
			image_id = anno_info['image_id'] - 1  # 0-based index.
			image_filename = annos['images'][image_id]['file_name']
			segmentations = anno_info['segmentation']  # All segmentations are of length 1. (x1, y1, x2, y2, ..., xn, yn).
			bbox = anno_info['bbox']  # (x, y, width, height).

			image_filepath = os.path.join(image_dir_path, image_filename)
			if prev_image_filename != image_filepath:
				img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

			for seg in segmentations:
				cv2.line(img, (seg[0], seg[1]), (seg[2], seg[3]), (0, 0, 255), 2, cv2.LINE_AA)
				cv2.line(img, (seg[2], seg[3]), (seg[4], seg[5]), (0, 0, 255), 2, cv2.LINE_AA)
				cv2.line(img, (seg[4], seg[5]), (seg[6], seg[7]), (0, 0, 255), 2, cv2.LINE_AA)
				cv2.line(img, (seg[6], seg[7]), (seg[0], seg[1]), (0, 0, 255), 2, cv2.LINE_AA)
			cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 1, cv2.LINE_AA)
			cv2.imshow('Image', img)
			cv2.waitKey(0)

			if idx >= 19:
				break
			prev_image_filename = image_filepath

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def get_tablebank_detection_dicts(data_dir_path):
	latex_image_dir_path = os.path.join(data_dir_path, 'Detection_data/Latex/images')
	latex_json_filepath = os.path.join(data_dir_path, 'Detection_data/Latex/Latex.json')
	word_image_dir_path = os.path.join(data_dir_path, 'Detection_data/Word/images')
	word_json_filepath = os.path.join(data_dir_path, 'Detection_data/Word/Word.json')

	with open(latex_json_filepath) as fd:
		latex_annos = json.load(fd)
	with open(word_json_filepath) as fd:
		word_annos = json.load(fd)

	latex_annos['annotations'] = sorted(latex_annos['annotations'], key=lambda item: item['image_id'])
	word_annos['annotations'] = sorted(word_annos['annotations'], key=lambda item: item['image_id'])

	dataset_dicts = []
	annos_lst = [latex_annos, word_annos]
	image_dir_path_lst = [latex_image_dir_path, word_image_dir_path]
	for annos, image_dir_path in zip(annos_lst, image_dir_path_lst):
		for idx, anno_info in enumerate(annos['annotations']):
			record = {}

			image_id = anno_info['image_id'] - 1  # 0-based index.
			image_filename = annos['images'][image_id]['file_name']
			image_filepath = os.path.join(image_dir_path, image_filename)
			height, width = annos['images'][image_id]['height'], annos['images'][image_id]['width']
			#height, width = cv2.imread(image_filepath).shape[:2]

			record['file_name'] = image_filepath
			record['image_id'] = image_id
			record['height'] = height
			record['width'] = width

			obj = {
				'bbox': anno_info['bbox'],
				'bbox_mode': detectron2.structures.BoxMode.XYWH_ABS,
				'segmentation': anno_info['segmentation'],
				'category_id': anno_info['category_id'],
				'iscrowd': anno_info['iscrowd']
			}
			record['annotations'] = [obj]
			dataset_dicts.append(record)

	return dataset_dicts

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def train_table_detector_using_detectron2(tablebank_data_dir_path):
	get_tablebank_detection_dicts(tablebank_data_dir_path)

	detectron2.data.DatasetCatalog.register('tablebank', lambda: get_tablebank_detection_dicts(tablebank_data_dir_path))
	detectron2.data.MetadataCatalog.get('tablebank').set(thing_classes=['table'])
	tablebank_metadata = detectron2.data.MetadataCatalog.get('tablebank')

	#--------------------
	# Fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the dataset.

	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
	cfg.DATASETS.TRAIN = ('tablebank',)
	cfg.DATASETS.TEST = ()
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo.
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.00025  # Pick a good LR.
	cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset.
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster, and good enough for this toy dataset (default: 512).
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only has one class (ballon).

	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = detectron2.engine.DefaultTrainer(cfg) 
	trainer.resume_or_load(resume=False)
	trainer.train()

	# Look at training curves in tensorboard:
	#	tensorboard --logdir output

	#--------------------
	# Infer using the trained model.

	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set the testing threshold for this model.
	cfg.DATASETS.TEST = ('tablebank',)
	predictor = detectron2.engine.DefaultPredictor(cfg)

	#--------------------
	# Randomly select several samples to visualize the prediction results.

	dataset_dicts = get_tablebank_detection_dicts(tablebank_data_dir_path)
	for d in random.sample(dataset_dicts, 3):    
		im = cv2.imread(d['file_name'])
		outputs = predictor(im)
		v = detectron2.utils.visualizer.Visualizer(im[:,:,::-1],
			metadata=tablebank_metadata, 
			scale=0.8, 
			instance_mode=detectron2.utils.visualizer.ColorMode.IMAGE_BW  # Remove the colors of unsegmented pixels.
		)
		v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
		cv2.imshow('Visualization', v.get_image()[:,:,::-1])
		cv2.waitKey(0)
	cv2.destroyAllWindows()

	#--------------------
	# Evaluate its performance using AP metric implemented in COCO API.

	evaluator = detectron2.evaluation.COCOEvaluator('tablebank', cfg, False, output_dir='./output/')
	val_loader = detectron2.data.build_detection_test_loader(cfg, 'tablebank')
	detectron2.evaluation.inference_on_dataset(trainer.model, val_loader, evaluator)

	# Another equivalent way is to use trainer.test.

def main():
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	tablebank_data_dir_path = data_base_dir_path + '/text/table_form/tablebank/TableBank_data'

	#check_tablebank_detection_data(tablebank_data_dir_path)
	#get_tablebank_detection_dicts(tablebank_data_dir_path)

	train_table_detector_using_detectron2(tablebank_data_dir_path)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
