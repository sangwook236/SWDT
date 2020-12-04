#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/facebookresearch/detectron2
#	https://detectron2.readthedocs.io/

import os, random, json
import numpy as np
#import torch, torchvision
import cv2
import detectron2
import detectron2.model_zoo, detectron2.config, detectron2.engine, detectron2.evaluation, detectron2.utils.visualizer, detectron2.data, detectron2.structures

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def simple_detection_example():
	image_filepath = './input.jpg'
	im = cv2.imread(image_filepath)
	if im is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	#cv2.imshow('Image', im)

	# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
	cfg = detectron2.config.get_cfg()
	# Add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library.
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model.
	# Find a model from detectron2's model zoo.
	# You can use the https://dl.fbaipublicfiles... url as well.
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')

	predictor = detectron2.engine.DefaultPredictor(cfg)
	outputs = predictor(im)

	# Look at the outputs.
	# See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification.
	print("outputs['instances'].pred_boxes = {}.".format(outputs['instances'].pred_boxes))  # A Boxes object storing N boxes, one for each detected instance.
	print("outputs['instances'].scores = {}.".format(outputs['instances'].scores))  # A vector of N scores.
	print("outputs['instances'].pred_classes = {}.".format(outputs['instances'].pred_classes))  # A vector of N labels in range [0, num_categories).
	print("outputs['instances'].pred_masks = {}.".format(outputs['instances'].pred_masks))  # A Tensor of shape (N, H, W), masks for each detected instance.

	# We can use 'Visualizer' to draw the predictions on the image.
	v = detectron2.utils.visualizer.Visualizer(im[:,:,::-1], detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
	cv2.imshow('Result', v.get_image()[:,:,::-1])

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def simple_keypoint_detection_example():
	image_filepath = './input.jpg'
	im = cv2.imread(image_filepath)
	if im is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	#cv2.imshow('Image', im)

	# Infer with a keypoint detection model.
	# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set threshold for this model.
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml')

	if True:
		# REF [site] >> https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
		# Annotations for keypoints:
		#	A number of keypoints is specified in sets of 3, (x, y, v).
		#	x and y indicate pixel positions in the image.
		#	v indicates visibility:
		#		v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.

		"""
		print('[Train dataset] Keys:', detectron2.data.DatasetCatalog.get(cfg.DATASETS.TRAIN[0])[0].keys())
		print('[Train dataset] #annotations:', len(detectron2.data.DatasetCatalog.get(cfg.DATASETS.TRAIN[0])[0]['annotations']))
		print("[Train dataset] Annotation's keys:", detectron2.data.DatasetCatalog.get(cfg.DATASETS.TRAIN[0])[0]['annotations'][0].keys())
		print("[Train dataset] Annotation's keypoints:", detectron2.data.DatasetCatalog.get(cfg.DATASETS.TRAIN[0])[0]['annotations'][0]['keypoints'])
		print('[Train dataset] Metadata:', detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
		"""

		print('[Test dataset] Keys:', detectron2.data.DatasetCatalog.get(cfg.DATASETS.TEST[0])[0].keys())
		print('[Test dataset] #annotations:', len(detectron2.data.DatasetCatalog.get(cfg.DATASETS.TEST[0])[0]['annotations']))
		print("[Test dataset] Annotation's keys:", detectron2.data.DatasetCatalog.get(cfg.DATASETS.TEST[0])[0]['annotations'][0].keys())
		print("[Test dataset] Annotation's keypoints:", detectron2.data.DatasetCatalog.get(cfg.DATASETS.TEST[0])[0]['annotations'][0]['keypoints'])
		print('[Test dataset] Metadata:', detectron2.data.MetadataCatalog.get(cfg.DATASETS.TEST[0]))

	predictor = detectron2.engine.DefaultPredictor(cfg)
	outputs = predictor(im)

	# Look at the outputs.
	# See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification.
	print("outputs['instances'].pred_boxes = {}.".format(outputs['instances'].pred_boxes))  # A Boxes object storing N boxes, one for each detected instance.
	print("outputs['instances'].scores = {}.".format(outputs['instances'].scores))  # A vector of N scores.
	print("outputs['instances'].pred_classes = {}.".format(outputs['instances'].pred_classes))  # A vector of N labels in range [0, num_categories).
	print("outputs['instances'].pred_masks = {}.".format(outputs['instances'].pred_masks))  # A Tensor of shape (N, H, W), masks for each detected instance.
	print("outputs['instances'].pred_keypoints = {}.".format(outputs['instances'].pred_keypoints))  # A Tensor of shape (N, num_keypoint, 3). Each row in the last dimension is (x, y, score). Scores are larger than 0.

	v = detectron2.utils.visualizer.Visualizer(im[:,:,::-1], detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
	cv2.imshow('Result', v.get_image()[:,:,::-1])

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def simple_panoptic_segmentation_example():
	image_filepath = './input.jpg'
	im = cv2.imread(image_filepath)
	if im is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	#cv2.imshow('Image', im)

	# Infer with a panoptic segmentation model.
	# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
	cfg = detectron2.config.get_cfg()
	cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'))
	cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml')

	predictor = detectron2.engine.DefaultPredictor(cfg)
	panoptic_seg, segments_info = predictor(im)['panoptic_seg']

	v = detectron2.utils.visualizer.Visualizer(im[:,:,::-1], detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	v = v.draw_panoptic_seg_predictions(panoptic_seg.to('cpu'), segments_info)
	cv2.imshow('Result', v.get_image()[:,:,::-1])

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# REF [site] >> https://detectron2.readthedocs.io/tutorials/datasets.html
def get_balloon_dicts(img_dir):
	json_file = os.path.join(img_dir, 'via_region_data.json')
	with open(json_file) as fd:
		imgs_anns = json.load(fd)

	dataset_dicts = []
	for idx, v in enumerate(imgs_anns.values()):
		record = {}
		
		filename = os.path.join(img_dir, v['filename'])
		height, width = cv2.imread(filename).shape[:2]
		
		record['file_name'] = filename
		record['image_id'] = idx
		record['height'] = height
		record['width'] = width
	  
		annos = v['regions']
		objs = []
		for _, anno in annos.items():
			assert not anno['region_attributes']
			anno = anno['shape_attributes']
			px = anno['all_points_x']
			py = anno['all_points_y']
			poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
			poly = [p for x in poly for p in x]

			obj = {
				'bbox': [np.min(px), np.min(py), np.max(px), np.max(py)],
				'bbox_mode': detectron2.structures.BoxMode.XYXY_ABS,
				'segmentation': [poly],
				'category_id': 0,
				'iscrowd': 0
			}
			objs.append(obj)
		record['annotations'] = objs
		dataset_dicts.append(record)
	return dataset_dicts

# REF [site] >> https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
def training_on_a_custom_dataset_example():
	classes = ['balloon']

	if False:
		# If your dataset is in COCO format:
		detectron2.data.datasets.register_coco_instances(name='balloon_train', metadata={'thing_classes': classes}, json_file='./balloon/train.json', image_root='./balloon/train')
		detectron2.data.datasets.register_coco_instances(name='balloon_val', metadata={'thing_classes': classes}, json_file='./balloon/val.json', image_root='./balloon/val')
	else:
		for tag in ['train', 'val']:
			data_name = 'balloon_{}'.format(tag)
			json_filepath = './balloon/{}.json'.format(tag)
			image_dir_path = './balloon/{}'.format(tag)
			detectron2.data.DatasetCatalog.register(data_name, lambda: get_balloon_dicts(image_dir_path))
			#detectron2.data.MetadataCatalog.get(data_name).set(evaluator_type='coco', json_file=json_filepath, image_root=image_dir_path, thing_classes=classes)
			detectron2.data.MetadataCatalog.get(data_name).set(thing_classes=classes)
	balloon_metadata = detectron2.data.MetadataCatalog.get('balloon_train')
	print('Balloon metadata = {}.'.format(balloon_metadata))

	#--------------------
	# Train.
	if True:
		# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
		cfg = detectron2.config.get_cfg()
		cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
		cfg.DATASETS.TRAIN = ('balloon_train',)
		cfg.DATASETS.TEST = ()
		cfg.DATALOADER.NUM_WORKERS = 2
		cfg.MODEL.WEIGHTS = detectron2.model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo.
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Faster, and good enough for this toy dataset (default: 512).
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # Only has one class (balloon).
		#cfg.MODEL.DEVICE = 'cuda:0'
		cfg.SOLVER.IMS_PER_BATCH = 2
		cfg.SOLVER.BASE_LR = 0.00025  # Pick a good LR.
		cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset.
		#cfg.OUTPUT_DIR = './output'  # Default output directory = './output'.

		os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

		#--------------------
		# Fine-tune a coco-pretrained R50-FPN Mask R-CNN model on the balloon dataset.
		trainer = detectron2.engine.DefaultTrainer(cfg)
		trainer.resume_or_load(resume=False)
		trainer.train()

		# Look at training curves in tensorboard:
		#	tensorboard --logdir output

		#--------------------
		# Evaluate its performance using AP metric implemented in COCO API.
		evaluator = detectron2.evaluation.COCOEvaluator('balloon_val', cfg, False, output_dir='./output')
		val_dataloader = detectron2.data.build_detection_test_loader(cfg, 'balloon_val')
		detectron2.evaluation.inference_on_dataset(trainer.model, val_dataloader, evaluator)

		# Another equivalent way is to use trainer.test.

	#--------------------
	# Evaluate its performance using AP metric implemented in COCO API.
	if True:
		# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
		cfg = detectron2.config.get_cfg()
		cfg.merge_from_file(detectron2.model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
		cfg.DATASETS.TEST = ('balloon_val',)
		cfg.DATALOADER.NUM_WORKERS = 8
		cfg.MODEL.WEIGHTS = './output/model_final.pth'
		cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

		model = detectron2.modeling.build_model(cfg)
		detectron2.checkpoint.DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

		evaluator = detectron2.evaluation.COCOEvaluator('balloon_val', cfg, False, output_dir='./output')
		val_dataloader = detectron2.data.build_detection_test_loader(cfg, 'balloon_val')
		detectron2.evaluation.inference_on_dataset(model, val_dataloader, evaluator)

	#--------------------
	# Infer using the trained model.
	if True:
		# REF [site] >> https://detectron2.readthedocs.io/modules/config.html
		cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # Set the testing threshold for this model.
		cfg.DATASETS.TEST = ('balloon_val',)

		predictor = detectron2.engine.DefaultPredictor(cfg)

		# Randomly select several samples to visualize the prediction results.
		dataset_dicts = get_balloon_dicts('./balloon/val')
		for dat in random.sample(dataset_dicts, 3):    
			im = cv2.imread(dat['file_name'])
			outputs = predictor(im)
			v = detectron2.utils.visualizer.Visualizer(im[:,:,::-1],
				metadata=balloon_metadata, 
				scale=0.8, 
				instance_mode=detectron2.utils.visualizer.ColorMode.IMAGE_BW  # Remove the colors of unsegmented pixels.
			)
			v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
			cv2.imshow('Visualization', v.get_image()[:,:,::-1])
			cv2.waitKey(0)
		cv2.destroyAllWindows()

def main():
	#simple_detection_example()
	#simple_keypoint_detection_example()
	#simple_panoptic_segmentation_example()

	training_on_a_custom_dataset_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
