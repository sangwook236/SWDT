#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torchvision
import cv2

def visualize_person_keypoints(img, predictions, BOX_SCORE_THRESHOLD):
	for prediction in predictions:
		#print('Prediction keys:', prediction.keys())
		print('#detected persons =', len(prediction['boxes'].detach().numpy()))
		print("#detected persons' scores =", prediction['scores'].detach().numpy())

		#for label, box, score, keypoints, keypoints_scores in zip(prediction['labels'], prediction['boxes'], prediction['scores'], prediction['keypoints'], predictions['keypoints_scores']):
		#	print(label.detach().numpy(), box.detach().numpy(), score.detach().numpy(), keypoints.detach().numpy(), keypoints_scores.detach().numpy())
		for box, score, keypoints in zip(prediction['boxes'], prediction['scores'], prediction['keypoints']):
			score = score.detach().numpy()
			if score < BOX_SCORE_THRESHOLD:
				continue

			box = box.detach().numpy()
			keypoints = keypoints.detach().numpy()[:,:2]

			assert len(keypoints) == 17

			cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[3]), tuple(keypoints[1]), (102, 204, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[4]), tuple(keypoints[2]), (51, 153, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[1]), tuple(keypoints[0]), (102, 0, 204), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[0]), tuple(keypoints[2]), (51, 102, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[5]), tuple(keypoints[6]), (255, 128, 0), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[5]), tuple(keypoints[7]), (153, 255, 204), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[6]), tuple(keypoints[8]), (128, 229, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[7]), tuple(keypoints[9]), (153, 255, 153), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[8]), tuple(keypoints[10]), (102, 255, 224), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[11]), tuple(keypoints[12]), (255, 102, 0), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[11]), tuple(keypoints[13]), (255, 255, 77), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[12]), tuple(keypoints[14]), (153, 255, 204), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[13]), tuple(keypoints[15]), (191, 255, 128), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[14]), tuple(keypoints[16]), (255, 195, 77), 2, cv2.LINE_AA)

		cv2.imwrite('person_keypoint_prediction.png', img)
		#cv2.imshow('Person Keypoint Prediction', img)
		#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def visualize_table_keypoints(img, predictions, BOX_SCORE_THRESHOLD):
	for prediction in predictions:
		#print('Prediction keys:', prediction.keys())
		print('#detected tables =', len(prediction['boxes'].detach().numpy()))
		print("#detected tables' scores =", prediction['scores'].detach().numpy())

		#for label, box, score, keypoints, keypoints_scores in zip(prediction['labels'], prediction['boxes'], prediction['scores'], prediction['keypoints'], predictions['keypoints_scores']):
		#	print(label.detach().numpy(), box.detach().numpy(), score.detach().numpy(), keypoints.detach().numpy(), keypoints_scores.detach().numpy())
		for box, score, keypoints in zip(prediction['boxes'], prediction['scores'], prediction['keypoints']):
			score = score.detach().numpy()
			if score < BOX_SCORE_THRESHOLD:
				continue

			box = box.detach().numpy()
			keypoints = keypoints.detach().numpy()[:,:2]

			assert len(keypoints) == 4

			#cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[0]), tuple(keypoints[1]), (0, 0, 255), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[1]), tuple(keypoints[2]), (0, 255, 0), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[2]), tuple(keypoints[3]), (255, 0, 0), 2, cv2.LINE_AA)
			cv2.line(img, tuple(keypoints[3]), tuple(keypoints[0]), (255, 0, 255), 2, cv2.LINE_AA)

		cv2.imwrite('table_keypoint_prediction.png', img)
		#cv2.imshow('Table Keypoint Prediction', img)
		#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def infer_person_keypoint_using_keypointrcnn_resnet50_fpn():
	#--------------------
	# Load an image.
	image_filepath = './input.jpg'
	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	#IMAGE_WIDTH = 480
	#img = cv2.resize(img, (IMAGE_WIDTH, int(img.height * IMAGE_WIDTH / img.width)))

	# Image to tensor.
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
	])

	input_tensors = [transform(img)]

	print("Input image's shape =", input_tensors[0].shape)

	#--------------------
	# Load a pretrained model.
	model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

	#--------------------
	# Infer.
	predictions = model(input_tensors)

	#--------------------
	# Visualize.
	BOX_SCORE_THRESHOLD = 0.9
	visualize_person_keypoints(img, predictions, BOX_SCORE_THRESHOLD)

	# Export the model to ONNX.
	#torch.onnx.export(model, input_tensors, './keypoint_rcnn.onnx', opset_version=11)

# REF [site] >> https://github.com/pytorch/vision/blob/master/torchvision/models/detection/keypoint_rcnn.py
def infer_person_keypoint_using_KeypointRCNN():
	# Our dataset has two classes only - background and person.
	num_classes = 2

	#--------------------
	# Load a pre-trained model for classification and return only the features.
	backbone = torchvision.models.mobilenet_v2(pretrained=True).features
	# KeypointRCNN needs to know the number of output channels in a backbone.
	# For mobilenet_v2, it's 1280 so we need to add it here.
	backbone.out_channels = 1280

	# Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
	# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios.
	anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
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

	keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
		featmap_names=[0],
		output_size=14,
		sampling_ratio=2
	)

	# Put the pieces together inside a KeypointRCNN model.
	model = torchvision.models.detection.KeypointRCNN(
		backbone,
		num_classes=num_classes,
		rpn_anchor_generator=anchor_generator,
		box_roi_pool=roi_pooler,
		keypoint_roi_pool=keypoint_roi_pooler
	)

	#--------------------
	# Train.

	# TODO [implement] >>

	#--------------------
	# Load an image.
	image_filepath = './input.jpg'
	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image: {}.'.format(image_filepath))
		return
	#IMAGE_WIDTH = 480
	#img = cv2.resize(img, (IMAGE_WIDTH, int(img.height * IMAGE_WIDTH / img.width)))

	# Image to tensor.
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
	])

	input_tensors = [transform(img)]

	print("Input image's shape =", input_tensors[0].shape)

	#--------------------
	# Infer.
	model.eval()
	predictions = model(input_tensors)

	#--------------------
	# Visualize.
	BOX_SCORE_THRESHOLD = 0.9
	visualize_person_keypoints(img, predictions, BOX_SCORE_THRESHOLD)

	# Export the model to ONNX.
	#torch.onnx.export(model, input_tensors, './keypoint_rcnn.onnx', opset_version=11)

def main():
	infer_person_keypoint_using_keypointrcnn_resnet50_fpn()
	#infer_person_keypoint_using_KeypointRCNN()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
