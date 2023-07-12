#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import requests
from PIL import Image
import torch
import transformers

# REF [site] >> https://huggingface.co/docs/transformers/main/model_doc/sam
def segment_anything_example():
	# Models:
	#	facebook/sam-vit-base.
	#	facebook/sam-vit-large: ~1.25GB.
	#	facebook/sam-vit-huge: ~2.56GB.

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if False:
		# Initializing a SamConfig with "facebook/sam-vit-huge" style configuration.
		configuration = transformers.SamConfig()

		# Initializing a SamModel (with random weights) from the "facebook/sam-vit-huge" style configuration.
		model = transformers.SamModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

		# We can also initialize a SamConfig from a SamVisionConfig, SamPromptEncoderConfig, and SamMaskDecoderConfig.

		# Initializing SAM vision, SAM Q-Former and language model configurations.
		vision_config = transformers.SamVisionConfig()
		prompt_encoder_config = transformers.SamPromptEncoderConfig()
		mask_decoder_config = transformers.SamMaskDecoderConfig()

		config = transformers.SamConfig(vision_config, prompt_encoder_config, mask_decoder_config)

	if True:
		model = transformers.SamModel.from_pretrained("facebook/sam-vit-large").to(device)
		processor = transformers.SamProcessor.from_pretrained("facebook/sam-vit-large")

		img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
		raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
		input_points = [[[450, 600]]]  # 2D location of a window in the image.
		inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)

		outputs = model(**inputs)

		masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
		scores = outputs.iou_scores
		print(f"Masks: shape = {masks[0].shape}, dtype = {masks[0].dtype}.")
		print(f"{scores=}.")

# REF [site] >> https://huggingface.co/nvidia
def segformer_example():
	# Models:
	#	nvidia/segformer-b0-finetuned-cityscapes-768-768.
	#	nvidia/segformer-b0-finetuned-cityscapes-512-1024.
	#	nvidia/segformer-b0-finetuned-cityscapes-640-1280.
	#	nvidia/segformer-b0-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b1-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b2-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b3-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b4-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b5-finetuned-cityscapes-1024-1024.
	#	nvidia/segformer-b0-finetuned-ade-512-512.
	#	nvidia/segformer-b1-finetuned-ade-512-512.
	#	nvidia/segformer-b2-finetuned-ade-512-512.
	#	nvidia/segformer-b3-finetuned-ade-512-512.
	#	nvidia/segformer-b4-finetuned-ade-512-512.
	#	nvidia/segformer-b5-finetuned-ade-640-640.

	feature_extractor = transformers.SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
	#processor = transformers.SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
	model = transformers.SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")

	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	inputs = feature_extractor(images=image, return_tensors="pt")
	#inputs = processor(images=image, return_tensors="pt")
	outputs = model(**inputs)
	logits = outputs.logits  # Shape: (batch_size, num_labels, height / 4, width / 4).
	print(f"{logits=}.")

def main():
	segment_anything_example()  # Segment Anything (SAM).

	#segformer_example()  # SegFormer.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
