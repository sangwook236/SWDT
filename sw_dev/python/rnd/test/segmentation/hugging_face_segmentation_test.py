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

def main():
	segment_anything_example()  # Segment Anything (SAM).

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
