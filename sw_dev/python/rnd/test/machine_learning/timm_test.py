#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import timm
import torch

# REF [site] >> https://huggingface.co/docs/timm/quickstart
def quickstart():
	# List models with pretrained weights.
	model_names = timm.list_models(pretrained=True)
	print(model_names)

	model_names = timm.list_models("*resne*t*")
	print(model_names)

	#-----
	# Load a pretrained model.
	model = timm.create_model("mobilenetv3_large_100", pretrained=True)
	model.eval()

	#-----
	# Fine-tune a pretrained model.
	model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=1000)

	# REF [site] >> https://huggingface.co/docs/timm/training_script.

	#-----
	# Use a pretrained model for feature extraction.
	x = torch.randn(1, 3, 224, 224)
	model = timm.create_model("mobilenetv3_large_100", pretrained=True)
	features = model.forward_features(x)
	print(features.shape)

	#-----
	# Image augmentation.
	print(timm.data.create_transform((3, 224, 224)))

	# To figure out which transformations were used for a given pretrained model.
	print(model.pretrained_cfg)

	# Resolve only the data related configuration by using timm.data.resolve_data_config().
	print(timm.data.resolve_data_config(model.pretrained_cfg))

	#-----
	# We can pass this data config to timm.data.create_transform() to initialize the model's associated transform.
	data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
	transform = timm.data.create_transform(**data_cfg)
	print(transform)

	# Using pretrained models for inference.
	import requests
	from PIL import Image

	url = "https://datasets-server.huggingface.co/assets/imagenet-1k/--/default/test/12/image/image.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	model = timm.create_model("mobilenetv3_large_100", pretrained=True).eval()
	transform = timm.data.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg))
	image_tensor = transform(image)
	print(image_tensor.shape)

	output = model(image_tensor.unsqueeze(0))
	print(output.shape)

	probabilities = torch.nn.functional.softmax(output[0], dim=0)
	print(probabilities.shape)

	values, indices = torch.topk(probabilities, 5)
	print(indices)

	IMAGENET_1k_URL = "https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt"
	IMAGENET_1k_LABELS = requests.get(IMAGENET_1k_URL).text.strip().split("\n")
	print([{"label": IMAGENET_1k_LABELS[idx], "value": val.item()} for val, idx in zip(values, indices)])

# REF [site] >> https://huggingface.co/docs/timm/feature_extraction
def feature_extraction_tutorial():
	# Unpooled.

	m = timm.create_model("xception41", pretrained=True)
	o = m(torch.randn(2, 3, 299, 299))
	print(f"Original shape: {o.shape}.")
	o = m.forward_features(torch.randn(2, 3, 299, 299))
	print(f"Unpooled shape: {o.shape}.")

	# Create with no classifier and pooling.
	m = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="")
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Unpooled shape: {o.shape}.")

	# Remove it later.
	m = timm.create_model("densenet121", pretrained=True)
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Original shape: {o.shape}.")
	m.reset_classifier(0, "")
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Unpooled shape: {o.shape}.")

	#-----
	# Pooled.

	# Create with no classifier.
	m = timm.create_model("resnet50", pretrained=True, num_classes=0)
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Pooled shape: {o.shape}.")

	# Remove it later.
	m = timm.create_model("ese_vovnet19b_dw", pretrained=True)
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Original shape: {o.shape}.")
	m.reset_classifier(0)
	o = m(torch.randn(2, 3, 224, 224))
	print(f"Pooled shape: {o.shape}.")

	#-----
	# Multi-scale feature maps (Feature pyramid).

	# Create a feature map extraction model.
	m = timm.create_model("resnest26d", features_only=True, pretrained=True)
	o = m(torch.randn(2, 3, 224, 224))
	for x in o:
		print(x.shape)

	# Query the feature information.
	m = timm.create_model("regnety_032", features_only=True, pretrained=True)
	print(f"Feature channels: {m.feature_info.channels()}.")
	o = m(torch.randn(2, 3, 224, 224))
	for x in o:
		print(x.shape)

	# Select specific feature levels or limit the stride.
	m = timm.create_model("ecaresnet101d", features_only=True, output_stride=8, out_indices=(2, 4), pretrained=True)
	print(f"Feature channels: {m.feature_info.channels()}.")
	print(f"Feature reduction: {m.feature_info.reduction()}.")
	o = m(torch.randn(2, 3, 320, 320))
	for x in o:
		print(x.shape)

# REF [site] >> https://huggingface.co/docs/timm/hf_hub
def huggingface_hub_tutorial():
	from huggingface_hub import notebook_login

	# Authenticating.
	notebook_login()

	# Sharing a model.
	model = timm.create_model("resnet18", pretrained=True, num_classes=4)

	# Loading a model.
	model_reloaded = timm.create_model("hf_hub:nateraw/resnet18-random", pretrained=True)

def main():
	#quickstart()

	#feature_extraction_tutorial()
	huggingface_hub_tutorial()

	#--------------------
	# Scripts.
	#	REF [site] >> https://huggingface.co/docs/timm/training_script
	#
	#	A train, validation, inference, and checkpoint cleaning script included in the github root folder.
	#	Scripts are not currently packaged in the pip release.

	# Training script.
	#	To train an SE-ResNet34 on ImageNet, locally distributed, 4 GPUs, one process per GPU w/ cosine schedule, random-erasing prob of 50% and per-pixel random value:
	#		./distributed_train.sh 4 /data/imagenet --model seresnet34 --sched cosine --epochs 150 --warmup-epochs 5 --lr 0.4 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 4

	# Validation / inference scripts.
	#	To validate with the model’s pretrained weights (if they exist):
	#		python validate.py /imagenet/validation/ --model seresnext26_32x4d --pretrained
	#	To run inference from a checkpoint:
	#		python inference.py /imagenet/validation/ --model mobilenetv3_large_100 --checkpoint ./output/train/model_best.pth.tar

	# Training examples.
	#	EfficientNet-B2 with RandAugment - 80.4 top-1, 95.1 top-5:
	#		./distributed_train.sh 2 /imagenet/ --model efficientnet_b2 -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .016
	#	MixNet-XL with RandAugment - 80.5 top-1, 94.9 top-5:
	#		./distributed_train.sh 2 /imagenet/ --model mixnet_xl -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .969 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.3 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.3 --amp --lr .016 --dist-bn reduce
	#	SE-ResNeXt-26-D and SE-ResNeXt-26-T:
	#		./distributed_train.sh 2 /imagenet/ --model seresnext26t_32x4d --lr 0.1 --warmup-epochs 5 --epochs 160 --weight-decay 1e-4 --sched cosine --reprob 0.4 --remode pixel -b 112
	#	EfficientNet-B3 with RandAugment - 81.5 top-1, 95.7 top-5:
	#	EfficientNet-B0 with RandAugment - 77.7 top-1, 95.3 top-5:
	#		./distributed_train.sh 2 /imagenet/ --model efficientnet_b0 -b 384 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .048
	#	ResNet50 with JSD loss and RandAugment (clean + 2x RA augs) - 79.04 top-1, 94.39 top-5:
	#		./distributed_train.sh 2 /imagenet -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce
	#	EfficientNet-ES (EdgeTPU-Small) with RandAugment - 78.066 top-1, 93.926 top-5:
	#		./distributed_train.sh 8 /imagenet --model efficientnet_es -b 128 --sched step --epochs 450 --decay-epochs 2.4 --decay-rate .97 --opt rmsproptf --opt-eps .001 -j 8 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2  --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064
	#	MobileNetV3-Large-100 - 75.766 top-1, 92,542 top-5:
	#		./distributed_train.sh 2 /imagenet/ --model mobilenetv3_large_100 -b 512 --sched step --epochs 600 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 7 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-path 0.2 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --amp --lr .064 --lr-noise 0.42 0.9
	#	ResNeXt-50 32x4d w/ RandAugment - 79.762 top-1, 94.60 top-5:
	#		./distributed_train.sh 8 /imagenet --model resnext50_32x4d --lr 0.6 --warmup-epochs 5 --epochs 240 --weight-decay 1e-4 --sched cosine --reprob 0.4 --recount 3 --remode pixel --aa rand-m7-mstd0.5-inc1 -b 192 -j 6 --amp --dist-bn reduce

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
