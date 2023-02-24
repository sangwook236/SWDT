#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, copy
import numpy as np
import mmdet, mmdet.models, mmdet.apis
import mmcv.runner
import torch, torchvision
import matplotlib.pyplot as plt

# REF [site] >> https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb
def detection_infer_demo():
	# REF [site] >> https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn
	# Choose to use a config and initialize the detector.
	config = "configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco.py"
	# Setup a checkpoint file to load.
	checkpoint = "checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth"

	# Set the device to be used for evaluation.
	device = "cuda:0"

	# Load the config.
	config = mmcv.Config.fromfile(config)
	# Set pretrained to be None since we do not need pretrained model here.
	config.model.pretrained = None

	# Initialize the detector.
	model = mmdet.models.build_detector(config.model)

	# Load checkpoint.
	checkpoint = mmcv.runner.load_checkpoint(model, checkpoint, map_location=device)

	# Set the classes of models for inference.
	model.CLASSES = checkpoint["meta"]["CLASSES"]

	# We need to set the model's cfg for inference.
	model.cfg = config

	# Convert the model to GPU.
	model.to(device)
	# Convert the model into evaluation mode.
	model.eval()

	#-----
	# Inference the detector.

	# Use the detector to do inference
	img = "demo/demo.jpg"
	result = mmdet.apis.inference_detector(model, img)

	# Let's plot the result.
	mmdet.apis.show_result_pyplot(model, img, result, score_thr=0.3)

# REF [site] >> https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb
def detection_train_demo():
	# Train a detector on a customized dataset.

	img = mmcv.imread("kitti_tiny/training/image_2/000073.jpeg")
	plt.figure(figsize=(15, 10))
	plt.imshow(mmcv.bgr2rgb(img))
	plt.show()

	@mmdet.datasets.builder.DATASETS.register_module()
	class KittiTinyDataset(mmdet.datasets.custom.CustomDataset):
		CLASSES = ("Car", "Pedestrian", "Cyclist")

		def load_annotations(self, ann_file):
			cat2label = {k: i for i, k in enumerate(self.CLASSES)}
			# Load image list from file.
			image_list = mmcv.list_from_file(self.ann_file)
		
			data_infos = []
			# Convert annotations to middle format.
			for image_id in image_list:
				filename = f"{self.img_prefix}/{image_id}.jpeg"
				image = mmcv.imread(filename)
				height, width = image.shape[:2]
		
				data_info = dict(filename=f"{image_id}.jpeg", width=width, height=height)
		
				# Load annotations.
				label_prefix = self.img_prefix.replace("image_2", "label_2")
				lines = mmcv.list_from_file(os.path.join(label_prefix, f"{image_id}.txt"))
		
				content = [line.strip().split(" ") for line in lines]
				bbox_names = [x[0] for x in content]
				bboxes = [[float(info) for info in x[4:8]] for x in content]
		
				gt_bboxes = []
				gt_labels = []
				gt_bboxes_ignore = []
				gt_labels_ignore = []
		
				# Filter "DontCare".
				for bbox_name, bbox in zip(bbox_names, bboxes):
					if bbox_name in cat2label:
						gt_labels.append(cat2label[bbox_name])
						gt_bboxes.append(bbox)
					else:
						gt_labels_ignore.append(-1)
						gt_bboxes_ignore.append(bbox)

				data_anno = dict(
					bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
					labels=np.array(gt_labels, dtype=np.long),
					bboxes_ignore=np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
					labels_ignore=np.array(gt_labels_ignore, dtype=np.long)
				)

				data_info.update(ann=data_anno)
				data_infos.append(data_info)

			return data_infos

	#-----
	# Modify the config.

	cfg = mmcv.Config.fromfile("./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py")

	# Modify dataset type and path.
	cfg.dataset_type = "KittiTinyDataset"
	cfg.data_root = "kitti_tiny/"

	cfg.data.test.type = "KittiTinyDataset"
	cfg.data.test.data_root = "kitti_tiny/"
	cfg.data.test.ann_file = "train.txt"
	cfg.data.test.img_prefix = "training/image_2"

	cfg.data.train.type = "KittiTinyDataset"
	cfg.data.train.data_root = "kitti_tiny/"
	cfg.data.train.ann_file = "train.txt"
	cfg.data.train.img_prefix = "training/image_2"

	cfg.data.val.type = "KittiTinyDataset"
	cfg.data.val.data_root = "kitti_tiny/"
	cfg.data.val.ann_file = "val.txt"
	cfg.data.val.img_prefix = "training/image_2"

	# Mmodify num classes of the model in box head.
	cfg.model.roi_head.bbox_head.num_classes = 3
	# If we need to finetune a model based on a pre-trained detector, we need to use load_from to set the path of checkpoints.
	cfg.load_from = "checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth"

	# Set up working dir to save files and logs.
	cfg.work_dir = "./tutorial_exps"

	# The original learning rate (LR) is set for 8-GPU training.
	# We divide it by 8 since we only use one GPU.
	cfg.optimizer.lr = 0.02 / 8
	cfg.lr_config.warmup = None
	cfg.log_config.interval = 10

	# Change the evaluation metric since we use customized dataset.
	cfg.evaluation.metric = "mAP"
	# We can set the evaluation interval to reduce the evaluation times.
	cfg.evaluation.interval = 12
	# We can set the checkpoint saving interval to reduce the storage cost.
	cfg.checkpoint_config.interval = 12

	# Set seed thus the results are more reproducible.
	cfg.seed = 0
	mmdet.apis.set_random_seed(0, deterministic=False)
	cfg.device = "cuda"
	cfg.gpu_ids = range(1)

	# We can also use tensorboard to log the training process.
	cfg.log_config.hooks = [
		dict(type="TextLoggerHook"),
		dict(type="TensorboardLoggerHook")
	]

	# We can initialize the logger for training and have a look at the final config used for training.
	print(f"Config:\n{cfg.pretty_text}")

	#-----
	# Train a new detector.

	# Build dataset.
	datasets = [mmdet.datasets.build_dataset(cfg.data.train)]

	# Build the detector.
	model = mmdet.models .build_detector(cfg.model)
	# Add an attribute for visualization convenience.
	model.CLASSES = datasets[0].CLASSES

	# Create work_dir.
	mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
	mmdet.apis.train_detector(model, datasets, cfg, distributed=False, validate=True)

	# Understand the log.
	#	Load tensorboard in colab:
	#		load_ext tensorboard
	#	See curves in tensorboard:
	#		tensorboard --logdir ./tutorial_exps

	#-----
	# Test the trained detector.

	img = mmcv.imread("kitti_tiny/training/image_2/000068.jpeg")

	model.cfg = cfg
	result = mmdet.apis.inference_detector(model, img)
	mmdet.apis.show_result_pyplot(model, img, result)

# REF [site] >> https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb
def instance_segmentation_infer_demo():
	# REF [site] >> https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn
	# Choose to use a config and initialize the detector.
	config = "configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py"
	# Setup a checkpoint file to load.
	checkpoint = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

	# Set the device to be used for evaluation.
	device = "cuda:0"

	# Load the config.
	config = mmcv.Config.fromfile(config)
	# Set pretrained to be None since we do not need pretrained model here.
	config.model.pretrained = None

	# Initialize the detector.
	model = mmdet.models.build_detector(config.model)

	# Load checkpoint.
	checkpoint = mmcv.runner.load_checkpoint(model, checkpoint, map_location=device)

	# Set the classes of models for inference.
	model.CLASSES = checkpoint["meta"]["CLASSES"]

	# We need to set the model's cfg for inference.
	model.cfg = config

	# Convert the model to GPU.
	model.to(device)
	# Convert the model into evaluation mode.
	model.eval()

	#-----
	# Inference with the detector.

	# Use the detector to do inference.
	img = "demo/demo.jpg"
	result = mmdet.apis.inference_detector(model, img)

	# Let's plot the result.
	mmdet.apis.show_result_pyplot(model, img, result, score_thr=0.3)

# REF [site] >> https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_InstanceSeg_Tutorial.ipynb
def instance_segmentation_train_demo():
	# Train a detector on a customized dataset.

	img = mmcv.imread("balloon/train/10464445726_6f1e3bbe6a_k.jpg")
	plt.figure(figsize=(15, 10))
	plt.imshow(mmcv.bgr2rgb(img))
	plt.show()

	# Check the label of a single image.
	annotation = mmcv.load("balloon/train/via_region_data.json")

	# The annotation is a dict, and its values looks like the following.
	annotation["34020010494_e5cb88e1c4_k.jpg1115004"]

	def convert_balloon_to_coco(ann_file, out_file, image_prefix):
		data_infos = mmcv.load(ann_file)

		annotations = []
		images = []
		obj_count = 0
		for idx, v in enumerate(mmcv.track_iter_progress(data_infos.values())):
			filename = v["filename"]
			img_path = os.path.join(image_prefix, filename)
			height, width = mmcv.imread(img_path).shape[:2]

			images.append(dict(
				id=idx,
				file_name=filename,
				height=height,
				width=width
			))

			bboxes = []
			labels = []
			masks = []
			for _, obj in v["regions"].items():
				assert not obj["region_attributes"]
				obj = obj["shape_attributes"]
				px = obj["all_points_x"]
				py = obj["all_points_y"]
				poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
				poly = [p for x in poly for p in x]

				x_min, y_min, x_max, y_max = (min(px), min(py), max(px), max(py))

				data_anno = dict(
					image_id=idx,
					id=obj_count,
					category_id=0,
					bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
					area=(x_max - x_min) * (y_max - y_min),
					segmentation=[poly],
					iscrowd=0
				)
				annotations.append(data_anno)
				obj_count += 1

		coco_format_json = dict(
			images=images,
			annotations=annotations,
			categories=[{"id":0, "name": "balloon"}]
		)
		mmcv.dump(coco_format_json, out_file)

	convert_balloon_to_coco(
		"balloon/train/via_region_data.json",
		"balloon/train/annotation_coco.json",
		"balloon/train/"
	)
	convert_balloon_to_coco(
		"balloon/val/via_region_data.json",
		"balloon/val/annotation_coco.json",
		"balloon/val/"
	)

	#-----
	# Modify the config.

	cfg = mmcv.Config.fromfile("./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py")

	# Modify dataset type and path.
	cfg.dataset_type = "COCODataset"

	cfg.data.test.ann_file = "balloon/val/annotation_coco.json"
	cfg.data.test.img_prefix = "balloon/val/"
	cfg.data.test.classes = ("balloon",)

	cfg.data.train.ann_file = "balloon/train/annotation_coco.json"
	cfg.data.train.img_prefix = "balloon/train/"
	cfg.data.train.classes = ("balloon",)

	cfg.data.val.ann_file = "balloon/val/annotation_coco.json"
	cfg.data.val.img_prefix = "balloon/val/"
	cfg.data.val.classes = ("balloon",)

	# Modify num classes of the model in box head and mask head.
	cfg.model.roi_head.bbox_head.num_classes = 1
	cfg.model.roi_head.mask_head.num_classes = 1

	# We can still the pre-trained Mask RCNN model to obtain a higher performance.
	cfg.load_from = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

	# Set up working dir to save files and logs.
	cfg.work_dir = "./tutorial_exps"

	# The original learning rate (LR) is set for 8-GPU training.
	# We divide it by 8 since we only use one GPU.
	cfg.optimizer.lr = 0.02 / 8
	cfg.lr_config.warmup = None
	cfg.log_config.interval = 10

	# We can set the evaluation interval to reduce the evaluation times.
	cfg.evaluation.interval = 12
	# We can set the checkpoint saving interval to reduce the storage cost.
	cfg.checkpoint_config.interval = 12

	# Set seed thus the results are more reproducible.
	cfg.seed = 0
	mmdet.apis.set_random_seed(0, deterministic=False)
	cfg.device = "cuda"
	cfg.gpu_ids = range(1)

	# We can also use tensorboard to log the training process.
	cfg.log_config.hooks = [
		dict(type="TextLoggerHook"),
		dict(type="TensorboardLoggerHook")
	]

	# We can initialize the logger for training and have a look at the final config used for training.
	print(f"Config:\n{cfg.pretty_text}")

	#-----
	# Train a new detector.

	# Build dataset.
	datasets = [mmdet.datasets.build_dataset(cfg.data.train)]

	# Build the detector.
	model = mmdet.models.build_detector(cfg.model)

	# Add an attribute for visualization convenience.
	model.CLASSES = datasets[0].CLASSES

	# Create work_dir.
	mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
	mmdet.apis.train_detector(model, datasets, cfg, distributed=False, validate=True)

	# Understand the log.
	#	Load tensorboard in colab:
	#		load_ext tensorboard
	#	See curves in tensorboard:
	#		tensorboard --logdir ./tutorial_exps

	#-----
	# Test the trained detector.

	img = mmcv.imread("balloon/train/7178882742_f090f3ce56_k.jpg")

	model.cfg = cfg
	result = mmdet.apis.inference_detector(model, img)
	mmdet.apis.show_result_pyplot(model, img, result)

def main():
	if True:
		import mmcv.ops

		print(torch.__version__, torch.cuda.is_available())
		print(mmdet.__version__)
		print(mmcv.ops.get_compiling_cuda_version())
		print(mmcv.ops.get_compiler_version())

		print(mmcv.collect_env())

	# Detection.
	#detection_infer_demo()
	#detection_train_demo()

	# Instance segmentation.
	#instance_segmentation_infer_demo()
	instance_segmentation_train_demo()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
