#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch, torchvision

def simple_classification_test():
	#model = torchvision.models.alexnet(pretrained=True)
	#model = torchvision.models.vgg16(pretrained=True)
	#model = torchvision.models.vgg16_bn(pretrained=True)
	model = torchvision.models.resnet50(pretrained=True)  # torchvision.models.resnet.ResNet.
	#model = torchvision.models.resnet101(pretrained=True)
	#model = torchvision.models.squeezenet1_1(pretrained=True)
	#model = torchvision.models.densenet121(pretrained=True)
	#model = torchvision.models.inception_v3(pretrained=True)
	#model = torchvision.models.googlenet(pretrained=True)
	#model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
	#model = torchvision.models.mobilenet_v2(pretrained=True)
	#model = torchvision.models.mobilenet_v3_large(pretrained=True)
	#model = torchvision.models.resnext50_32x4d(pretrained=True)
	#model = torchvision.models.wide_resnet50_2(pretrained=True)
	#model = torchvision.models.mnasnet1_0(pretrained=True)
	#model = torchvision.models.efficientnet_b0(pretrained=True)
	#model = torchvision.models.regnet_y_400mf(pretrained=True)
	#model = torchvision.models.vit_b_16(pretrained=True)
	#model = torchvision.models.convnext_base(pretrained=True)
	#print(model)

	print("Modules:")
	for name, module in model._modules.items():
		# Layer names: transform, backbone, rpn, roi_heads.
		print("\t{}: {}.".format(name, type(module)))

	#--------------------
	if True:
		mean = [0.485, 0.456, 0.406]
		stddev = [0.229, 0.224, 0.225]
	else:
		transform = torchvision.transforms.Compose([
			torchvision.transforms.Resize(256),
			torchvision.transforms.CenterCrop(224),
			torchvision.transforms.ToTensor()
		])
		dataset = torchvision.datasets.ImageNet(".", split="train", transform=transform)

		means, stds = list(), list()
		for img in torchvision.subset(dataset):
			means.append(torch.mean(img))
			stds.append(torch.std(img))

		mean = torch.mean(torch.tensor(means))
		stddev = torch.mean(torch.tensor(stds))

	normalize = torchvision.transforms.Normalize(mean=mean, std=stddev)

	#--------------------
	# Inference.
	model.eval()

	x = torch.rand(5, 3, 224, 224)
	#x = torch.rand(7, 3, 360, 120)
	#x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]  # Error.
	x = normalize(x)

	predictions = model(x)
	print("predictions: shape = {}, dtype = {}.".format(predictions.shape, predictions.dtype))

	predictions = torch.argmax(predictions, dim=-1)
	print("predictions: shape = {}, dtype = {}.".format(predictions.shape, predictions.dtype))

	#--------------------
	# Export the model to ONNX.
	#torch.onnx.export(model, x, "./imagenet_classification.onnx", opset_version=11)

# REF [site] >> https://pytorch.org/vision/stable/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
def fasterrcnn_resnet50_fpn_test():
	# torchvision.models.detection.faster_rcnn.FasterRCNN.
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, num_classes=91, trainable_backbone_layers=5)
	#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT, num_classes=91, trainable_backbone_layers=5)
	#model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, num_classes=91, trainable_backbone_layers=5)
	#print(model)

	# Module names: transform, backbone, rpn, roi_heads.
	print("Modules:")
	for name, module in model._modules.items():
		print("\t{}: {}.".format(name, type(module)))

	# REF [function] >> resnet_fpn_backbone_test()
	backbone = model._modules["backbone"]  # torchvision.models.detection.backbone_utils.BackboneWithFPN.

	#--------------------
	# Training.
	images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
	boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
	labels = torch.randint(1, 91, (4, 11))
	images = list(image for image in images)
	targets = list()
	for i in range(len(images)):
		d = {}
		d["boxes"] = boxes[i]
		d["labels"] = labels[i]
		targets.append(d)
	output = model(images, targets)

	#--------------------
	# Inference.
	model.eval()
	x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
	predictions = model(x)

	print("#predictions = {}.".format(len(predictions)))
	for idx, prediction in enumerate(predictions):
		print("Prediction {}:".format(idx))
		print("\t{}: {}.".format("labels", prediction["labels"]))
		print("\t{}: {}.".format("boxes", prediction["boxes"]))
		print("\t{}: {}.".format("scores", prediction["scores"]))

	#--------------------
	# Export the model to ONNX.
	#torch.onnx.export(model, x, "./faster_rcnn.onnx", opset_version=11)

# REF [site] >> https://pytorch.org/vision/stable/generated/torchvision.models.detection.fcos_resnet50_fpn.html
def fcos_resnet50_fpn_test():
	# torchvision.models.detection.fcos.FCOS.
	model = torchvision.models.detection.fcos_resnet50_fpn(pretrained=True, num_classes=91, trainable_backbone_layers=5)
	#print(model)

	# Module names: backbone, anchor_generator, head, transform.
	print("Modules:")
	for name, module in model._modules.items():
		print("\t{}: {}.".format(name, type(module)))

	# REF [function] >> resnet_fpn_backbone_test()
	backbone = model._modules["backbone"]  # torchvision.models.detection.backbone_utils.BackboneWithFPN.

	#--------------------
	# Inference.
	model.eval()
	x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
	predictions = model(x)

	print("#predictions = {}.".format(len(predictions)))
	for idx, prediction in enumerate(predictions):
		print("Prediction {}:".format(idx))
		print("\t{}: {}.".format("labels", prediction["labels"]))
		print("\t{}: {}.".format("boxes", prediction["boxes"]))
		print("\t{}: {}.".format("scores", prediction["scores"]))

# REF [site] >> https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
def resnet_fpn_backbone_test():
	from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

	# ResNet Paper		| torchvision.models.ResNet		| torchvision.models.detection.backbone_utils.BackboneWithFPN
	#-----------------------------------------------------------------------------------------------------------------
	# conv1		1/2		| conv1 + bn1 + relu + maxpool	|
	# conv2		1/4		| layer1						| 0
	# conv3		1/8		| layer2						| 1
	# conv4		1/16	| layer3						| 2
	# conv5		1/32	| layer4						| 3

	# torchvision.models.detection.backbone_utils.BackboneWithFPN.
	backbone = resnet_fpn_backbone("resnet50", pretrained=True, trainable_layers=5)
	#backbone = resnet_fpn_backbone("resnet50", weights=ResNet50_Weights.DEFAULT, trainable_layers=5)
	#print(backbone)

	# Get some dummy image.
	x = torch.rand(1, 3, 64, 64)
	#x = torch.rand(5, 3, 224, 27)
	#x = [torch.rand(3, 64, 64), torch.rand(3, 128, 128)]  # Error.

	# Compute the output.
	backbone_outputs = backbone(x)
	print([(name, outp.shape) for name, outp in backbone_outputs.items()])

def main():
	# REF [site] >> https://pytorch.org/vision/stable/models.html

	#--------------------
	# Classification.
	#	AlexNet, VGG, ResNet, SqueezeNet, DenseNet, Inception v3, GoogLeNet, ShuffleNet v2, MobileNetV2, MobileNetV3, ResNeXt, Wide ResNet, MNASNet, EfficientNet, RegNet, VisionTransformer, ConvNeXt.

	#simple_classification_test()

	#--------------------
	# Quantized models.
	#	MobileNetV2, MobileNetV3, ShuffleNet v2, ResNet, Inception v3, GoogLeNet.

	#--------------------
	# Semantic segmentation.
	#	FCN ResNet, DeepLabV3 ResNet, DeepLabV3 MobileNetV3, LR-ASPP MobileNetV3.

	#--------------------
	# Object detection, instance segmentation, and person keypoint detection.
	#	Faster R-CNN, FCOS, Mask R-CNN, RetinaNet, SSD, SSDlite.

	# REF [file] >>
	#	./pytorch_object_detection.py
	#	./pytorch_keypoint_detection.py

	#fasterrcnn_resnet50_fpn_test()
	#fcos_resnet50_fpn_test()

	resnet_fpn_backbone_test()

	#--------------------
	# Video classification.
	#	ResNet 3D, ResNet MC, ResNet (2+1)D.

	#--------------------
	# Optical flow.
	#	Raft.

#---------------------------------------------------------------------

if "__main__" == __name__:
	main()
