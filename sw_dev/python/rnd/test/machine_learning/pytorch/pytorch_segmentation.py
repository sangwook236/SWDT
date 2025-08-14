#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt

# REF [site] >> https://docs.pytorch.org/vision/main/models.html
def semantic_segmentation_example():
	#from torchvision.io.image import decode_image
	#from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
	#from torchvision.transforms.functional import to_pil_image

	# REF [site] >> https://github.com/pytorch/vision/tree/main/gallery/assets
	image_filepath = "../../../machine_learning/dog1.jpg"
	#image_filepath = "../../../machine_learning/dog2.jpg"

	#img = torchvision.io.decode_image(image_filepath)  # Runtime error
	img = torchvision.io.read_image(image_filepath)

	# Step 1: Initialize model with the best available weights
	if False:
		weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
	elif False:
		weights = torchvision.models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.deeplabv3_resnet101(weights=weights)
	elif True:
		weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
		#weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
	elif False:
		weights = torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
		#weights = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.fcn_resnet101(weights=weights)
	elif False:
		weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
	elif False:
		weights = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
	model.eval()

	# Step 2: Initialize the inference transforms
	preprocess = weights.transforms()

	# Step 3: Apply inference preprocessing transforms
	batch = preprocess(img).unsqueeze(0)

	# Step 4: Use the model and visualize the prediction
	prediction = model(batch)["out"]
	normalized_masks = prediction.softmax(dim=1)
	print(f"normalized_masks: shape = {normalized_masks.shape}, type = {normalized_masks.dtype}, (min, max) = ({normalized_masks.min().item()}, {normalized_masks.max().item()}).")

	class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
	mask = normalized_masks[0, class_to_idx["dog"]]
	print(f"mask: shape = {mask.shape}, type = {mask.dtype}, (min, max) = ({mask.min().item()}, {mask.max().item()}).")

	torchvision.transforms.functional.to_pil_image(mask).show()

# REF [site] >>
#	https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html
#	https://docs.pytorch.org/vision/main/models.html
def semantic_segmentation_visualization_example():
	plt.rcParams["savefig.bbox"] = "tight"

	def show(imgs):
		if not isinstance(imgs, list):
			imgs = [imgs]
		fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
		for i, img in enumerate(imgs):
			img = img.detach()
			img = torchvision.transforms.functional.to_pil_image(img)
			axs[0, i].imshow(np.asarray(img))
			axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	# REF [site] >> https://github.com/pytorch/vision/tree/main/gallery/assets
	#dog1_int = torchvision.io.decode_image(str(Path("../../../machine_learning") / "dog1.jpg"))
	#dog2_int = torchvision.io.decode_image(str(Path("../../../machine_learning") / "dog2.jpg"))
	dog1_int = torchvision.io.read_image(str(Path("../../../machine_learning") / "dog1.jpg"))
	dog2_int = torchvision.io.read_image(str(Path("../../../machine_learning") / "dog2.jpg"))
	input_image_list = [dog1_int, dog2_int]
	print(f"input_image_list[0]: shape = {input_image_list[0].shape}, dtype = {input_image_list[0].dtype}, (min, max) = ({input_image_list[0].min()}, {input_image_list[0].max()}).")

	if False:
		weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
		#weights = torchvision.models.segmentation.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.fcn_resnet50(weights=weights)
	else:
		weights = torchvision.models.segmentation.FCN_ResNet101_Weights.DEFAULT
		#weights = torchvision.models.segmentation.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
		model = torchvision.models.segmentation.fcn_resnet101(weights=weights)
	model = model.eval()
	transforms = weights.transforms(resize_size=None)

	images = torch.stack([transforms(d) for d in input_image_list])
	print(f"images[0]: shape = {images[0].shape}, dtype = {images[0].dtype}, (min, max) = ({images[0].min()}, {images[0].max()}).")

	output = model(images)["out"]
	print(f"Output: shape = {output.shape}, dtype = {output.dtype}, (min, max) = ({output.min().item()}, {output.max().item()}).")

	# Plot the masks
	sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
	normalized_masks = torch.nn.functional.softmax(output, dim=1)

	dog_and_boat_masks = [
		normalized_masks[img_idx, sem_class_to_idx[cls]]
		for img_idx in range(len(input_image_list))
		for cls in ("dog", "boat")
	]
	show(dog_and_boat_masks)

	# Get boolean masks
	class_dim = 1
	boolean_dog_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx["dog"])
	print(f"boolean_dog_masks: shape = {boolean_dog_masks.shape}, dtype = {boolean_dog_masks.dtype}, (min, max) = ({boolean_dog_masks.min().item()}, {boolean_dog_masks.max().item()}).")
	show([m.float() for m in boolean_dog_masks])

	# Plot boolean masks on top of the original images
	dogs_with_masks = [
		torchvision.utils.draw_segmentation_masks(img, masks=mask, alpha=0.7)
		for img, mask in zip(input_image_list, boolean_dog_masks)
	]
	show(dogs_with_masks)

	# For each pixel and each class C, is class C the most likely class?
	num_classes = normalized_masks.shape[1]
	img1_masks = normalized_masks[0]
	class_dim = 0
	img1_all_classes_masks = img1_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None]

	print(f"img1_masks: shape = {img1_masks.shape}, dtype = {img1_masks.dtype}, (min, max) = ({img1_masks.min().item()}, {img1_masks.max().item()}).")
	print(f"img1_all_classes_masks: shape = {img1_all_classes_masks.shape}, dtype = {img1_all_classes_masks.dtype}, (min, max) = ({img1_all_classes_masks.min().item()}, {img1_all_classes_masks.max().item()}).")

	img_with_all_masks = torchvision.utils.draw_segmentation_masks(input_image_list[0], masks=img1_all_classes_masks, alpha=.6)
	show(img_with_all_masks)

	class_dim = 1
	all_classes_masks = normalized_masks.argmax(class_dim) == torch.arange(num_classes)[:, None, None, None]
	print(f"all_classes_masks: shape = {all_classes_masks.shape}, dtype = {all_classes_masks.dtype}, (min, max) = ({all_classes_masks.min().item()}, {all_classes_masks.max().item()}).")
	# The first dimension is the classes now, so we need to swap it
	all_classes_masks = all_classes_masks.swapaxes(0, 1)

	imgs_with_masks = [
		torchvision.utils.draw_segmentation_masks(img, masks=mask, alpha=.6)
		for img, mask in zip(input_image_list, all_classes_masks)
	]
	show(imgs_with_masks)

	plt.show()

# REF [site] >>
#	https://docs.pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html
#	https://docs.pytorch.org/vision/main/models.html
def instance_segmentation_visualization_example():
	plt.rcParams["savefig.bbox"] = "tight"

	def show(imgs):
		if not isinstance(imgs, list):
			imgs = [imgs]
		fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
		for i, img in enumerate(imgs):
			img = img.detach()
			img = torchvision.transforms.functional.to_pil_image(img)
			axs[0, i].imshow(np.asarray(img))
			axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	# REF [site] >> https://github.com/pytorch/vision/tree/main/gallery/assets
	#dog1_int = torchvision.io.decode_image(str(Path("../../../machine_learning") / "dog1.jpg"))
	#dog2_int = torchvision.io.decode_image(str(Path("../../../machine_learning") / "dog2.jpg"))
	dog1_int = torchvision.io.read_image(str(Path("../../../machine_learning") / "dog1.jpg"))
	dog2_int = torchvision.io.read_image(str(Path("../../../machine_learning") / "dog2.jpg"))
	input_image_list = [dog1_int, dog2_int]
	print(f"input_image_list[0]: shape = {input_image_list[0].shape}, dtype = {input_image_list[0].dtype}, (min, max) = ({input_image_list[0].min()}, {input_image_list[0].max()}).")

	if False:
		weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
		#weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
		model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights, progress=False)
	else:
		weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
		#weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
		model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
	model = model.eval()
	transforms = weights.transforms()

	images = [transforms(d) for d in input_image_list]
	print(f"images[0]: shape = {images[0].shape}, dtype = {images[0].dtype}, (min, max) = ({images[0].min()}, {images[0].max()}).")

	output = model(images)
	print(f"Output: keys = {output[0].keys()}.")  # ['boxes', 'labels', 'scores', 'masks']
	print(f'\tBoxes: shape = {output[0]["boxes"].shape}, dtype = {output[0]["boxes"].dtype}, (min, max) = ({output[0]["boxes"].min()}, {output[0]["boxes"].max()}).')
	print(f'\tLabels: shape = {output[0]["labels"].shape}, dtype = {output[0]["labels"].dtype}, (min, max) = ({output[0]["labels"].min()}, {output[0]["labels"].max()}).')
	print(f'\tScores: shape = {output[0]["scores"].shape}, dtype = {output[0]["scores"].dtype}, (min, max) = ({output[0]["scores"].min()}, {output[0]["scores"].max()}).')
	print(f'\tMasks: shape = {output[0]["masks"].shape}, dtype = {output[0]["masks"].dtype}, (min, max) = ({output[0]["masks"].min()}, {output[0]["masks"].max()}).')

	# The boxes can be plotted with draw_bounding_boxes()

	# Masks
	img1_output = output[0]
	img1_masks = img1_output["masks"]
	print(f"img1_masks: shape = {img1_masks.shape}, dtype = {img1_masks.dtype}, (min, max) = ({img1_masks.min()}, {img1_masks.max()}).")

	print("For the first dog, the following instances were detected:")
	print([weights.meta["categories"][label] for label in img1_output["labels"]])

	# Boolean masks
	proba_threshold = 0.5
	img1_bool_masks = img1_output["masks"] > proba_threshold
	print(f"img1_bool_masks: shape = {img1_bool_masks.shape}, dtype = {img1_bool_masks.dtype}, (min, max) = ({img1_bool_masks.min()}, {img1_bool_masks.max()}).")

	# There's an extra dimension (1) to the masks. We need to remove it
	img1_bool_masks = img1_bool_masks.squeeze(1)

	show(torchvision.utils.draw_segmentation_masks(input_image_list[0], img1_bool_masks, alpha=0.9))
	print(f'Scores = {img1_output["scores"].detach().numpy()}.')

	# Plot the masks that have a good score
	score_threshold = .75
	boolean_masks = [
		out["masks"][out["scores"] > score_threshold] > proba_threshold
		for out in output
	]

	imgs_with_masks = [
		torchvision.utils.draw_segmentation_masks(img, mask.squeeze(1))
		for img, mask in zip(input_image_list, boolean_masks)
	]
	show(imgs_with_masks)

	plt.show()

def main():
	# Semantic segmentation
	#	REF [site] >> https://github.com/pytorch/vision/tree/main/references/segmentation
	#
	#	FCN ResNet50
	#	FCN ResNet101
	#	DeepLabv3 ResNet50
	#	DeepLabv3 ResNet101
	#	DeepLabv3 MobileNetV3-Large
	#	LR-ASPP MobileNetV3-Large

	# REF [file] >> ./pytorch_model.py

	semantic_segmentation_example()  # DeepLab, FCN, LR-ASPP
	#semantic_segmentation_visualization_example()  # FCN

	# Instance segmentation
	#	REF [file] >> ./pytorch_object_detection.py

	#instance_segmentation_visualization_example()  # Mask R-CNN

	# Object detection & instance segmentation
	#	REF [file] >> ./pytorch_object_detection.py
	# Keypoint detection
	#	REF [file] >> ./pytorch_keypoint_detection.py

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
