#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torchvision
import cv2

def visualize_person_keypoints(images, predictions, BOX_SCORE_THRESHOLD):
	for idx, (img, prediction) in enumerate(zip(images, predictions)):
		#print('Prediction keys: {}.'.format(prediction.keys()))
		print('#detected persons = {}.'.format(len(prediction['boxes'].detach().cpu().numpy())))
		print("#detected persons' scores = {}.".format(prediction['scores'].detach().cpu().numpy()))

		#for label, box, score, keypoints, keypoints_scores in zip(prediction['labels'], prediction['boxes'], prediction['scores'], prediction['keypoints'], predictions['keypoints_scores']):
		#	print(label.detach().cpu().numpy(), box.detach().cpu().numpy(), score.detach().cpu().numpy(), keypoints.detach().cpu().numpy(), keypoints_scores.detach().cpu().numpy())
		for box, score, keypoints in zip(prediction['boxes'], prediction['scores'], prediction['keypoints']):
			score = score.detach().cpu().numpy()
			if score < BOX_SCORE_THRESHOLD:
				continue

			box = box.detach().cpu().numpy().astype(np.int32)
			keypoints = keypoints.detach().cpu().numpy()[:,:2].astype(np.int32)

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

		cv2.imwrite('person_keypoint_prediction_{}.png'.format(idx), img)
		#cv2.imshow('Person Keypoint Prediction', img)
		#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def detect_person_keypoints():
	# Load an image.
	image_filepath = '/path/to/image.png'
	img = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
	if img is None:
		print('Failed to load an image from {}.'.format(image_filepath))
		return
	#IMAGE_WIDTH = 480
	#img = cv2.resize(img, (IMAGE_WIDTH, int(img.height * IMAGE_WIDTH / img.width)))

	print("Input image's shape = {}.".format(img.shape))

	# Image to tensor.
	transform = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
	])

	input_tensors = [transform(img)]

	#--------------------
	# Load a pretrained model.
	model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

	#--------------------
	# Infer.
	with torch.no_grad():
		predictions = model(input_tensors)

	# Keys:
	#	Keypoint detection: {boxes, labels, scores, keypoints, keypoints_scores}.
	#		boxes: [#detections, 4]. [x1, y1, x2, y2]. 0 <= x1 < x2 <= W & 0 <= y1 < y2 <= H.
	#		labels: [#detections].
	#		scores: [#detections].
	#		keypoints: [#detections, #keypoints, 3]. [x, y, visibility].
	#		keypoints_scores: [#detections, #keypoints].
	print("Prediction's keys = {}.".format(predictions[0].keys()))

	#--------------------
	# Visualize.
	BOX_SCORE_THRESHOLD = 0.9
	visualize_person_keypoints([img], predictions, BOX_SCORE_THRESHOLD)

	# Export the model to ONNX.
	#torch.onnx.export(model, input_tensors, './person_keypoint_rcnn_resnet50.onnx', opset_version=11)

# REF [function] >> rcnn_torchvision_tutorial() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_object_detection.py
def train_person_keypoints():
	# REF [site] >> https://github.com/pytorch/vision/tree/master/references/detection
	import torchvision_detection.transforms, torchvision_detection.engine

	num_classes = 2  # Our dataset has two classes only - background and person.
	num_keypoints = 17  # The number of person keypoints.
	num_epochs = 10

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
		model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

		dataset = PersonKeypointDataset(..., get_transform(train=True))
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
	# Create datasets.

	# FIXME [implement] >>
	train_dataset = PersonKeypointDataset(..., get_transform(train=True))
	test_dataset = PersonKeypointDataset(..., get_transform(train=False))

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
	# Create a model.
	if False:
		# REF [site] >> https://github.com/pytorch/vision/blob/main/torchvision/models/detection/keypoint_rcnn.py

		# Load a pre-trained model for classification and return only the features.
		backbone = torchvision.models.mobilenet_v2(pretrained=True).features
		# KeypointRCNN needs to know the number of output channels in a backbone.
		# For mobilenet_v2, it's 1280, so we need to add it here.
		backbone.out_channels = 1280

		# Let's make the RPN generate 5 x 3 anchors per spatial location, with 5 different sizes and 3 different aspect ratios.
		# We have a Tuple[Tuple[int]] because each feature map could potentially have different sizes and aspect ratios
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

		keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
			featmap_names=['0'],
			output_size=14,
			sampling_ratio=2,
		)

		# Put the pieces together inside a KeypointRCNN model.
		model = torchvision.models.detection.KeypointRCNN(
			backbone,
			num_classes=num_classes,
			rpn_anchor_generator=anchor_generator,
			box_roi_pool=roi_pooler,
			keypoint_roi_pool=keypoint_roi_pooler,
		)
	else:
		# REF [site] >> https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

		# Load a pre-trained model.
		model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)

		# Get the number of input features for the classifier.
		in_features = model.roi_heads.box_predictor.cls_score.in_features
		# Replace the pre-trained head with a new one.
		model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_channels=in_features, num_classes=num_classes)

		# TODO [check] >>
		# Get the number of input features for the keypoint classifier.
		in_features_keypoint = model.roi_heads.keypoint_predictor.input_features
		# Replace the keypoint predictor with a new one.
		model.roi_heads.keypoint_predictor = torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(in_channels=in_features_keypoint, num_keypoints=num_keypoints)

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

	model.eval()
	with torch.no_grad():
		predictions = model([img.to(device) for img in images])

	# Keys:
	#	Keypoint detection: {boxes, labels, scores, keypoints, keypoints_scores}.
	#		boxes: [#detections, 4]. [x1, y1, x2, y2]. 0 <= x1 < x2 <= W & 0 <= y1 < y2 <= H.
	#		labels: [#detections].
	#		scores: [#detections].
	#		keypoints: [#detections, #keypoints, 3]. [x, y, visibility].
	#		keypoints_scores: [#detections, #keypoints].
	print("Prediction's keys = {}.".format(predictions[0].keys()))

	#-----
	# Visualize.
	BOX_SCORE_THRESHOLD = 0.9
	visualize_person_keypoints([img.mul(255).permute(1, 2, 0).byte().numpy() for img in images], predictions, BOX_SCORE_THRESHOLD)

	# Export the model to ONNX.
	#torch.onnx.export(model, input_tensors, './person_keypoint_rcnn_mobilenet.onnx', opset_version=11)

def main():
	# Keypoint detection.
	detect_person_keypoints()
	#train_person_keypoints()  # Not yet completed.

	# Object detection & instance segmentation.
	#	REF [file] >> ./pytorch_object_detection.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
