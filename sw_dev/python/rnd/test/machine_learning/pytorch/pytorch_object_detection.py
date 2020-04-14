#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
#import cv2

# REF [site] >> https://github.com/pytorch/vision/tree/master/references/detection
import transforms as T
import utils, engine

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
		boxes = []
		for i in range(num_objs):
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])

		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		# There is only one class.
		labels = torch.ones((num_objs,), dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)

		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		# Suppose all instances are not crowd.
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

		target = {}
		target['boxes'] = boxes
		target['labels'] = labels
		target['masks'] = masks
		target['image_id'] = image_id
		target['area'] = area
		target['iscrowd'] = iscrowd
		#target['keypoints'] = keypoints

		if self.transforms is not None:
			img, target = self.transforms(img, target)

		return img, target

	def __len__(self):
		return len(self.imgs)

def create_object_detection_model(num_classes):
	if False:
		# Load a pre-trained model for classification and return only the features.
		backbone = torchvision.models.mobilenet_v2(pretrained=True).features
		# FasterRCNN needs to know the number of output channels in a backbone.
		# For mobilenet_v2, it's 1280 so we need to add it here.
		backbone.out_channels = 1280

		# Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
		# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios.
		anchor_generator = torchvision.models.detection.rpn. AnchorGenerator(
			sizes=((32, 64, 128, 256, 512),),
			aspect_ratios=((0.5, 1.0, 2.0),)
		)

		# Let's define what are the feature maps that we will use to perform the region of interest cropping, as well as the size of the crop after rescaling.
		# If your backbone returns a Tensor, featmap_names is expected to be [0].
		# More generally, the backbone should return an OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
		roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=[0],
			output_size=7,
			sampling_ratio=2
		)

		# Put the pieces together inside a FasterRCNN model.
		model = torchvision.models.detection.FasterRCNN(
			backbone,
			num_classes=num_classes,
			rpn_anchor_generator=anchor_generator,
			box_roi_pool=roi_pooler
		)
	else:
		# Load a model pre-trained pre-trained on COCO.
		model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	# Get number of input features for the classifier.
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# Replace the pre-trained head with a new one.
	model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

	return model

def create_instance_segmentation_model(num_classes):
	# Load an instance segmentation model pre-trained on COCO.
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

	# Get the number of input features for the classifier.
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# Replace the pre-trained head with a new one.
	model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

	# Get the number of input features for the mask classifier.
	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layer = 256
	# Replace the mask predictor with a new one.
	model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

	return model

def get_transform(train):
	transforms = []
	# Converts the image, a PIL image, into a PyTorch Tensor.
	transforms.append(T.ToTensor())
	if train:
		# During training, randomly flip the training images and ground-truth for data augmentation.
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)

def visualize_object_detection(img, predictions, BOX_SCORE_THRESHOLD):
	img = Image.fromarray(img)
	draw = ImageDraw.Draw(img)

	for prediction in predictions:
		mask_img = Image.new('L', img.size, 0)
		for scores, boxes in zip(prediction['scores'], prediction['boxes']):
			scores = scores.cpu().numpy()
			boxes = boxes.cpu().numpy()

			if scores.ndim == 0:
				if scores < BOX_SCORE_THRESHOLD:
					continue

				draw.rectangle(tuple(boxes), fill=None, outline=(255, 0, 0, 255))
			else:
				for idx, (score, box) in enumerate(zip(scores, boxes)):
					if score < BOX_SCORE_THRESHOLD:
						continue

					draw.rectangle(tuple(box), fill=None, outline=(255, 0, 0, 255))

		img.save('./PennFudanPed_bbox.png')
		#img.show()

def visualize_instance_segmentation(img, predictions, BOX_SCORE_THRESHOLD):
	img = Image.fromarray(img)
	draw = ImageDraw.Draw(img)

	for prediction in predictions:
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
				for idx, (score, box, mask) in enumerate(zip(scores, boxes, masks)):
					if score < BOX_SCORE_THRESHOLD:
						continue

					draw.rectangle(tuple(box), fill=None, outline=(255, 0, 0, 255))
					mask_img.paste(Image.new('L', mask.size, idx), (0, 0), Image.fromarray(mask[0], 'L'))

		img.save('./PennFudanPed_bbox.png')
		#img.show()
		mask_img.putpalette([
			0, 0, 0,  # Black background.
			255, 0, 0,  # Index 1 is red.
			255, 255, 0,  # Index 2 is yellow.
			255, 153, 0,  # Index 3 is orange.
		])
		mask_img.save('./PennFudanPed_mask.png')
		#mask_img.show()

# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def object_detection_finetuning_tutorial(is_instance_segmentation=True):
	if 'posix' == os.name:
		data_base_dir_path = '/home/sangwook/work/dataset'
	else:
		data_base_dir_path = 'D:/work/dataset'
	data_dir_path = data_base_dir_path + '/PennFudanPed'

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
	# Create a Dataset object.
	dataset = PennFudanDataset(data_dir_path, get_transform(train=True))
	dataset_test = PennFudanDataset(data_dir_path, get_transform(train=False))

	# Split the dataset in train and test set.
	torch.manual_seed(1)
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# Define training and validation data loaders.
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4,
		collate_fn=utils.collate_fn)

	#--------------------
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# Our dataset has two classes only - background and person.
	num_classes = 2

	#--------------------
	# Create an instance segmentation model for PennFudan dataset.
	if is_instance_segmentation:
		model = create_instance_segmentation_model(num_classes)
	else:
		model = create_object_detection_model(num_classes)
	model.to(device)

	if True:
		#--------------------
		# Construct an optimizer.
		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

		# A learning rate scheduler which decreases the learning rate by 10x every 3 epochs.
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

		#--------------------
		# Train it for 10 epochs.
		num_epochs = 10
		for epoch in range(num_epochs):
			# Train for one epoch, printing every 10 iterations.
			engine.train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

			# Update the learning rate.
			lr_scheduler.step()

			# Evaluate on the test dataset.
			engine.evaluate(model, data_loader_test, device=device)

	#--------------------
	# Infer
	img, _ = dataset_test[0]

	# Put the model in evaluation mode.
	model.eval()
	with torch.no_grad():
		predictions = model([img.to(device)])

	print("Prediction's keys:", predictions[0].keys())

	#--------------------
	# Visualize
	BOX_SCORE_THRESHOLD = 0.9
	if is_instance_segmentation:
		visualize_instance_segmentation(img.mul(255).permute(1, 2, 0).byte().numpy(), predictions, BOX_SCORE_THRESHOLD)
	else:
		visualize_object_detection(img.mul(255).permute(1, 2, 0).byte().numpy(), predictions, BOX_SCORE_THRESHOLD)

def main():
	object_detection_finetuning_tutorial(is_instance_segmentation=True)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
