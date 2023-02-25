#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
#import cv2

# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
class PennFudanDataset(torch.utils.data.Dataset):
	def __init__(self, root, transforms=None):
		self.root = root
		self.transforms = transforms
		# Load all image files, sorting them to ensure that they are aligned.
		self.imgs = list(sorted(os.listdir(os.path.join(root, 'PNGImages'))))
		self.masks = list(sorted(os.listdir(os.path.join(root, 'PedMasks'))))

	def __getitem__(self, idx):
		# Load images ad masks.
		img_path = os.path.join(self.root, 'PNGImages', self.imgs[idx])
		mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])
		img = Image.open(img_path).convert('RGB')
		# Note that we haven't converted the mask to RGB,
		# because each color corresponds to a different instance with 0 being background.
		mask = Image.open(mask_path)

		mask = np.array(mask)
		# Instances are encoded as different colors.
		obj_ids = np.unique(mask)
		# First id is the background, so remove it.
		obj_ids = obj_ids[1:]

		# Split the color-encoded mask into a set of binary masks.
		masks = mask == obj_ids[:, None, None]

		# Get bounding box coordinates for each mask.
		num_objs = len(obj_ids)
		boxes = list()
		for i in range(num_objs):
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])

		image_id = torch.tensor([idx])
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((num_objs,), dtype=torch.int64)  # Only one class.
		areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowds = torch.zeros((num_objs,), dtype=torch.int64)  # Suppose all instances are not crowd.
		masks = torch.as_tensor(masks, dtype=torch.uint8)

		target = {
			'image_id': image_id,
			'boxes': boxes,  # [#detections, 4]. [x1, y1, x2, y2]. 0 <= x1 < x2 <= W & 0 <= y1 < y2 <= H.
			'labels': labels,  # [#detections].
			'area': areas,  # [#detections].
			'iscrowd': iscrowds,  # [#detections].
			'masks': masks,  # For instance segmentation. [#detections, H, W]. Binary mask.
			#'keypoints': keypoints,  # For keypoint detection. [#detections, #keypoints, 3]. [x, y, visibility].
		}

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.imgs)

# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def create_object_detection_model(num_classes):
	if False:
		# REF [site] >>
		#	https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
		#	https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py

		# Load a pre-trained model for classification and return only the features.
		backbone = torchvision.models.mobilenet_v2(pretrained=True).features
		# FasterRCNN needs to know the number of output channels in a backbone.
		# For mobilenet_v2, it's 1280, so we need to add it here.
		backbone.out_channels = 1280

		# Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
		# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios.
		anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
			sizes=((32, 64, 128, 256, 512),),
			aspect_ratios=((0.5, 1.0, 2.0),),
		)

		# Let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling.
		# If your backbone returns a Tensor, featmap_names is expected to be ['0'].
		# More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=['0'],
			output_size=7,
			sampling_ratio=2,
		)

		# Put the pieces together inside a FasterRCNN model.
		model = torchvision.models.detection.FasterRCNN(
			backbone,
			num_classes=num_classes,
			rpn_anchor_generator=anchor_generator,
			box_roi_pool=roi_pooler,
		)
	else:
		# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

		# Load a model pre-trained on COCO.
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

		# Get the number of input features for the classifier.
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		# Replace the pre-trained head with a new one.
		model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)

	return model

# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def create_instance_segmentation_model(num_classes):
	if False:
		# REF [site] >> https://github.com/pytorch/vision/blob/main/torchvision/models/detection/mask_rcnn.py

		# Load a pre-trained model for classification and return only the features.
		backbone = torchvision.models.mobilenet_v2(pretrained=True).features
		# MaskRCNN needs to know the number of output channels in a backbone.
		# For mobilenet_v2, it's 1280 so we need to add it here,
		backbone.out_channels = 1280

		# Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
		# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios.
		anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
			sizes=((32, 64, 128, 256, 512),),
			aspect_ratios=((0.5, 1.0, 2.0),),
		)

		# Let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling.
		# If your backbone returns a Tensor, featmap_names is expected to be ['0'].
		# More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=['0'],
			output_size=7,
			sampling_ratio=2,
		)

		mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=['0'],
			output_size=14,
			sampling_ratio=2,
		)

		# Put the pieces together inside a MaskRCNN model.
		model = torchvision.models.detection.MaskRCNN(
			backbone,
			num_classes=num_classes,
			rpn_anchor_generator=anchor_generator,
			box_roi_pool=roi_pooler,
			mask_roi_pool=mask_roi_pooler,
		)
	else:
		# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

		# Load an instance segmentation model pre-trained on COCO.
		model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

		# Get the number of input features for the classifier.
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		# replace the pre-trained head with a new one
		model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)

		# Get the number of input features for the mask classifier.
		in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
		hidden_layer = 256
		# Replace the mask predictor with a new one.
		model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=hidden_layer, num_classes=num_classes)

	return model

def visualize_object_detection(images, predictions, BOX_SCORE_THRESHOLD):
	for idx, (img, prediction) in enumerate(zip(images, predictions)):
		img = Image.fromarray(img)
		draw = ImageDraw.Draw(img)

		#mask_img = Image.new('L', img.size, 0)
		for scores, boxes in zip(prediction['scores'], prediction['boxes']):
			scores = scores.cpu().numpy()
			boxes = boxes.cpu().numpy()

			if scores.ndim == 0:
				if scores < BOX_SCORE_THRESHOLD:
					continue

				draw.rectangle(tuple(boxes), fill=None, outline=(255, 0, 0, 255))
			else:
				for score, box in zip(scores, boxes):
					if score < BOX_SCORE_THRESHOLD:
						continue

					draw.rectangle(tuple(box), fill=None, outline=(255, 0, 0, 255))

		img.save('./PennFudanPed_bbox_{}.png'.format(idx))
		#img.show()

def visualize_instance_segmentation(images, predictions, BOX_SCORE_THRESHOLD):
	for idx, (img, prediction) in enumerate(zip(images, predictions)):
		img = Image.fromarray(img)
		draw = ImageDraw.Draw(img)

		mask_img = Image.new('L', img.size, 0)
		for scores, boxes, masks in zip(prediction['scores'], prediction['boxes'], prediction['masks']):
			scores = scores.cpu().numpy()
			boxes = boxes.cpu().numpy()
			masks = masks.mul(255).byte().cpu().numpy()

			if scores.ndim == 0:
				if scores < BOX_SCORE_THRESHOLD:
					continue

				draw.rectangle(tuple(boxes), fill=None, outline=(255, 0, 0, 255))
				mask_img.paste(Image.new('L', mask_img.size, 1), (0, 0), Image.fromarray(masks[0], 'L'))
			else:
				for ii, (score, box, mask) in enumerate(zip(scores, boxes, masks)):
					if score < BOX_SCORE_THRESHOLD:
						continue

					draw.rectangle(tuple(box), fill=None, outline=(255, 0, 0, 255))
					mask_img.paste(Image.new('L', mask.size, ii), (0, 0), Image.fromarray(mask[0], 'L'))

		img.save('./PennFudanPed_bbox_{}.png'.format(idx))
		#img.show()
		mask_img.putpalette([
			0, 0, 0,  # Black background.
			255, 0, 0,  # Index 1 is red.
			255, 255, 0,  # Index 2 is yellow.
			255, 153, 0,  # Index 3 is orange.
		])
		mask_img.save('./PennFudanPed_mask_{}.png'.format(idx))
		#mask_img.show()

# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def rcnn_torchvision_tutorial(is_instance_segmentation=True):
	# REF [site] >> https://github.com/pytorch/vision/tree/master/references/detection
	import torchvision_detection.transforms, torchvision_detection.engine

	num_classes = 2  # Our dataset has two classes only - background and person.
	num_epochs = 10

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	#--------------------
	data_dir_path = './PennFudanPed'

	# Check images and masks.
	if False:
		img = Image.open(data_dir_path + '/PNGImages/FudanPed00001.png')
		mask = Image.open(data_dir_path + '/PedMasks/FudanPed00001_mask.png')
		mask.putpalette([
			0, 0, 0,  # Black background.
			255, 0, 0,  # Index 1 is red.
			255, 255, 0,  # Index 2 is yellow.
			255, 153, 0,  # Index 3 is orange.
		])

		#img.save('./PennFudanPed_img.png')
		#mask.save('./PennFudanPed_mask.png')
		img.show()
		mask.show()

	#--------------------
	def get_transform(train):
		transforms = []
		# Converts the image, a PIL image, into a PyTorch Tensor.
		transforms.append(torchvision_detection.transforms.ToTensor())
		if train:
			# During training, randomly flip the training images and ground-truth for data augmentation.
			transforms.append(torchvision_detection.transforms.RandomHorizontalFlip(0.5))
		return torchvision_detection.transforms.Compose(transforms)

	# REF [function] >> collate_fn() in https://github.com/pytorch/vision/tree/master/references/detection/utils.py
	def collate_fn(batch):
		return tuple(zip(*batch))

	#--------------------
	if False:
		if is_instance_segmentation:
			model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
		else:
			model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

		dataset = PennFudanDataset(data_dir_path, get_transform(train=True))
		data_loader = torch.utils.data.DataLoader(
			dataset, batch_size=2, shuffle=True, num_workers=4,
			collate_fn=collate_fn
		)

		# For training.
		images, targets = next(iter(data_loader))
		images = list(image for image in images)
		targets = [{k: v for k, v in t.items()} for t in targets]
		output = model(images, targets)  # Returns losses and detections.

		# For inference.
		model.eval()
		x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
		predictions = model(x)  # Returns predictions.

	#--------------------
	# Create Dataset objects.
	train_dataset = PennFudanDataset(data_dir_path, get_transform(train=True))
	test_dataset = PennFudanDataset(data_dir_path, get_transform(train=False))

	# Split the dataset in train and test set.
	torch.manual_seed(1)
	indices = torch.randperm(len(train_dataset)).tolist()
	train_dataset = torch.utils.data.Subset(train_dataset, indices[:-50])
	test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

	# Define training and validation data loaders.
	train_dataloader = torch.utils.data.DataLoader(
		train_dataset, batch_size=2, shuffle=True, num_workers=4,
		collate_fn=collate_fn
	)
	test_dataloader = torch.utils.data.DataLoader(
		test_dataset, batch_size=1, shuffle=False, num_workers=4,
		collate_fn=collate_fn
	)

	#--------------------
	# Create a model for PennFudan dataset.
	if is_instance_segmentation:
		model = create_instance_segmentation_model(num_classes)
	else:
		model = create_object_detection_model(num_classes)
	model.to(device)

	#--------------------
	# Train.
	if True:
		# Construct an optimizer.
		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

		# A learning rate scheduler which decreases the learning rate by 10x every 3 epochs.
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

		for epoch in range(num_epochs):
			# Train for one epoch, printing every 10 iterations.
			torchvision_detection.engine.train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)

			# Update the learning rate.
			lr_scheduler.step()

			# Evaluate on the test dataset.
			torchvision_detection.engine.evaluate(model, test_dataloader, device=device)  # Person keypoints are evaluated by default when the IoU type is 'keypoints'.

	#--------------------
	# Infer.
	images = [
		test_dataset[0][0],
		test_dataset[1][0],
		test_dataset[2][0],
	]

	# Put the model in evaluation mode.
	model.eval()
	with torch.no_grad():
		predictions = model([img.to(device) for img in images])

	# Keys:
	#	Object detection: {boxes, labels, scores}.
	#	Instance segmentation: {boxes, labels, scores, masks}.
	#		boxes: [#detections, 4]. [x1, y1, x2, y2]. 0 <= x1 < x2 <= W & 0 <= y1 < y2 <= H.
	#		labels: [#detections].
	#		scores: [#detections].
	#		masks: [#detections, 1, H, W]. [0, 1].
	print("Prediction's keys = {}.".format(predictions[0].keys()))

	#-----
	# Visualize.
	BOX_SCORE_THRESHOLD = 0.9
	if is_instance_segmentation:
		visualize_instance_segmentation([img.mul(255).permute(1, 2, 0).byte().numpy() for img in images], predictions, BOX_SCORE_THRESHOLD)
	else:
		visualize_object_detection([img.mul(255).permute(1, 2, 0).byte().numpy() for img in images], predictions, BOX_SCORE_THRESHOLD)

def main():
	# Object detection & instance segmentation.
	rcnn_torchvision_tutorial(is_instance_segmentation=True)

	# Keypoint detection.
	#	REF [file] >> ./pytorch_keypoint_detection.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
