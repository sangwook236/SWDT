#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
def dinov2_example():
	import torch
	import transformers

	if True:
		pipe = transformers.pipeline(
			task="image-classification",
			model="facebook/dinov2-small-imagenet1k-1-layer",
			dtype=torch.float16,
			device=0
		)

		pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")

	if True:
		# Install
		#	pip install torchao

		import requests
		import torchao
		from PIL import Image

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-giant-imagenet1k-1-layer")

		quant_config = torchao.quantizcation.Int4WeightOnlyConfig(group_size=128)
		quantization_config = transformers.TorchAoConfig(quant_type=quant_config)

		model = transformers.AutoModelForImageClassification.from_pretrained(
			"facebook/dinov2-giant-imagenet1k-1-layer",
			dtype=torch.bfloat16,
			device_map="auto",
			quantization_config=quantization_config
		)

		inputs = processor(images=image, return_tensors="pt")
		outputs = model(**inputs)
		logits = outputs.logits
		predicted_class_idx = logits.argmax(-1).item()
		print("Predicted class:", model.config.id2label[predicted_class_idx])

	if True:
		import requests
		from PIL import Image

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		print(image.height, image.width)  # [480, 640]

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-base")
		model = transformers.AutoModel.from_pretrained("facebook/dinov2-base")
		patch_size = model.config.patch_size

		inputs = processor(images=image, return_tensors="pt")
		print(inputs.pixel_values.shape)  # [1, 3, 224, 224]
		batch_size, rgb, img_height, img_width = inputs.pixel_values.shape
		num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
		num_patches_flat = num_patches_height * num_patches_width

		outputs = model(**inputs)
		last_hidden_states = outputs[0]
		print(last_hidden_states.shape)  # [1, 1 + 256, 768]
		assert last_hidden_states.shape == (batch_size, 1 + num_patches_flat, model.config.hidden_size)

		cls_token = last_hidden_states[:, 0, :]
		patch_features = last_hidden_states[:, 1:, :].unflatten(1, (num_patches_height, num_patches_width))

	if True:
		import requests
		from PIL import Image

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov2-base")
		model = transformers.AutoModel.from_pretrained("facebook/dinov2-base")

		inputs = processor(images=image, return_tensors="pt")
		outputs = model(**inputs)
		last_hidden_states = outputs[0]

		# We have to force return_dict=False for tracing
		model.config.return_dict = False

		with torch.no_grad():
			traced_model = torch.jit.trace(model, [inputs.pixel_values])
			traced_outputs = traced_model(inputs.pixel_values)

		print((last_hidden_states - traced_outputs[0]).abs().max())

	if False:
		# Initializing a Dinov2 dinov2-base-patch16-224 style configuration
		configuration = transformers.Dinov2Config()

		# Initializing a model (with random weights) from the dinov2-base-patch16-224 style configuration
		model = transformers.Dinov2Model(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		import datasets

		dataset = datasets.load_dataset("huggingface/cats-image")
		image = dataset["test"]["image"][0]

		image_processor = transformers.AutoImageProcessor.from_pretrained("google/dinov2-base-patch16-224")
		model = transformers.Dinov2ForImageClassification.from_pretrained("google/dinov2-base-patch16-224")

		inputs = image_processor(image, return_tensors="pt")

		with torch.no_grad():
			logits = model(**inputs).logits

		# model predicts one of the 1000 ImageNet classes
		predicted_label = logits.argmax(-1).item()
		print(model.config.id2label[predicted_label])

# REF [site] >> https://huggingface.co/docs/transformers/main/en/model_doc/dinov3
def dinov3_example():
	import torch
	import transformers

	if True:
		pipe = transformers.pipeline(
			task="image-feature-extraction",
			model="facebook/dinov3-vits16-pretrain-lvd1689m",
			dtype=torch.bfloat16,
		)

		pipe("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")

	if True:
		# Install:
		#	pip install torchao

		from torchao.quantization import Int4WeightOnlyConfig

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = transformers.image_utils.load_image(url)

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov3-vitsplus-pretrain-lvd1689m")

		quant_type = Int4WeightOnlyConfig(group_size=128)
		quantization_config = transformers.TorchAoConfig(quant_type=quant_type)

		model = transformers.AutoModel.from_pretrained(
			"facebook/dinov3-vit7b16-pretrain-lvd1689m",
			dtype=torch.bfloat16,
			device_map="auto",
			quantization_config=quantization_config
		)

		inputs = processor(images=image, return_tensors="pt").to(model.device)
		with torch.inference_mode():
			outputs = model(**inputs)

		pooled_output = outputs.pooler_output
		print("Pooled output shape:", pooled_output.shape)

	if True:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = transformers.image_utils.load_image(url)
		print("Image size:", image.height, image.width)  # [480, 640]

		processor = transformers.AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
		model = transformers.AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
		patch_size = model.config.patch_size
		print("Patch size:", patch_size) # 16
		print("Num register tokens:", model.config.num_register_tokens)  # 4

		inputs = processor(images=image, return_tensors="pt")
		print("Preprocessed image size:", inputs.pixel_values.shape)  # [1, 3, 224, 224]

		batch_size, _, img_height, img_width = inputs.pixel_values.shape
		num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
		num_patches_flat = num_patches_height * num_patches_width

		with torch.inference_mode():
			outputs = model(**inputs)

		last_hidden_states = outputs.last_hidden_state
		print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
		assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

		cls_token = last_hidden_states[:, 0, :]
		patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :]
		patch_features = patch_features_flat.unflatten(1, (num_patches_height, num_patches_width))

	if False:
		# Initializing a DINOv3 ViT-small style configuration
		config = transformers.DINOv3ViTConfig()

		# Initializing a model (with random weights) from the config
		model = transformers.DINOv3ViTModel(config)

		# Accessing the model config
		config = model.config

	if False:
		# Initializing a DINOv3ConvNext (tiny variant) style configuration
		config = transformers.DINOv3ConvNextConfig()

		# Initializing a model (with random weights)
		model = transformers.DINOv3ConvNextModel(config)

		# Accessing the model config
		config = model.config

def main():
	# Models:
	#	Barlow Twins
	#	BYOL
	#	CMC
	#	MoCo
	#	ReLIC
	#	SimCLR
	#	SimSiam
	#	SwAV
	#
	#	DINOv1, DINOv2, DINOv3
	#	DeepCluster
	#	CLIP

	# Self-DIstillation with NO labels (DINO)
	#dinov2_example()
	dinov3_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
