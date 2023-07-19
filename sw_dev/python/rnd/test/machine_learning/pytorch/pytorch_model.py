#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch, torchvision

# REF [site] >> https://pytorch.org/vision/stable/models.html
def pre_trained_models_example():
	from torchvision.models import resnet50, ResNet50_Weights

	# Initializing pre-trained models.

	# Old weights with accuracy 76.130%
	resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

	# New weights with accuracy 80.858%
	resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

	# Best available weights (currently alias for IMAGENET1K_V2)
	# Note that these weights may change across versions
	resnet50(weights=ResNet50_Weights.DEFAULT)

	# Strings are also supported
	resnet50(weights="IMAGENET1K_V2")

	# No weights - random initialization
	resnet50(weights=None)

	# Using pretrained weights:
	resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
	resnet50(weights="IMAGENET1K_V1")
	resnet50(pretrained=True)  # Deprecated
	resnet50(True)  # Deprecated

	# Using no weights:
	resnet50(weights=None)
	resnet50()
	resnet50(pretrained=False)  # Deprecated
	resnet50(False)  # Deprecated

	#-----
	# Using the pre-trained models.

	# Initialize the Weight Transforms
	weights = ResNet50_Weights.DEFAULT
	preprocess = weights.transforms()

	# Apply it to the input image
	img = torch.rand(5, 3, 512, 512)
	img_transformed = preprocess(img)

	# Initialize model
	model = resnet50(weights=weights)

	# Set model to eval mode
	model.eval()
	with torch.no_grad():
		outputs = model(img_transformed)

	#-----
	# Listing and retrieving available models.

	from torchvision.models import get_model, get_model_weights, get_weight, list_models

	# List available models
	all_models = list_models()
	classification_models = list_models(module=torchvision.models)

	# Initialize models
	m1 = get_model("mobilenet_v3_large", weights=None)
	m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

	# Fetch weights
	weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
	assert weights == torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights.DEFAULT

	weights_enum = get_model_weights("quantized_mobilenet_v3_large")
	assert weights_enum == torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights

	weights_enum2 = get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
	assert weights_enum == weights_enum2

	#-----
	# Using models from Hub.

	# Option 1: passing weights param as string
	model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

	# Option 2: passing weights param as enum
	weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
	model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)

	weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
	print([weight for weight in weight_enum])

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

	pre_trained_models_example()

	# For ViT models:
	#	REF [function] >> vit_test() in ./pytorch_transformer.py

	#--------------------
	# Classification.
	#	AlexNet, VGG, ResNet, SqueezeNet, DenseNet, Inception v3, GoogLeNet, ShuffleNet v2, MobileNetV2, MobileNetV3, ResNeXt, Wide ResNet, MNASNet, EfficientNet, RegNet, VisionTransformer, ConvNeXt.

	# REF [file] >> ./pytorch_classification.py

	#simple_classification_test()

	#--------------------
	# Quantized models.
	#	MobileNetV2, MobileNetV3, ShuffleNet v2, ResNet, Inception v3, GoogLeNet.

	#--------------------
	# Semantic segmentation.
	#	FCN ResNet, DeepLabV3 ResNet, DeepLabV3 MobileNetV3, LR-ASPP MobileNetV3.

	# REF [file] >> ./pytorch_segmentation.py

	#--------------------
	# Object detection, instance segmentation, and person keypoint detection.
	#	Faster R-CNN, FCOS, Mask R-CNN, RetinaNet, SSD, SSDlite.

	# REF [file] >>
	#	./pytorch_object_detection.py
	#	./pytorch_keypoint_detection.py

	#fasterrcnn_resnet50_fpn_test()
	#fcos_resnet50_fpn_test()

	#resnet_fpn_backbone_test()

	#--------------------
	# Video classification.
	#	ResNet 3D, ResNet MC, ResNet (2+1)D.

	#--------------------
	# Optical flow.
	#	Raft.

#---------------------------------------------------------------------

if "__main__" == __name__:
	main()
