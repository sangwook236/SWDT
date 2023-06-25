#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import requests
import torch
from PIL import Image
import transformers

# REF [site] >> https://huggingface.co/docs/transformers/model_doc/owlvit
def owl_vit_test():
	if False:
		# Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration.
		configuration = transformers.OwlViTTextConfig()

		# Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration.
		model = transformers.OwlViTTextModel(configuration)

		# Accessing the model configuration
		configuration = model.config

	if False:
		# Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration
		configuration = transformers.OwlViTVisionConfig()

		# Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration.
		model = transformers.OwlViTVisionModel(configuration)

		# Accessing the model configuration.
		configuration = model.config

	if False:
		model = transformers.OwlViTModel.from_pretrained("google/owlvit-base-patch32")
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = processor(text=[["a photo of a cat", "a photo of a dog"]], images=image, return_tensors="pt")

		outputs = model(**inputs)

		logits_per_image = outputs.logits_per_image  # This is the image-text similarity score.
		probs = logits_per_image.softmax(dim=1)  # We can take the softmax to get the label probabilities.

	if False:
		model = transformers.OwlViTModel.from_pretrained("google/owlvit-base-patch32")
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")

		inputs = processor(text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt")

		text_features = model.get_text_features(**inputs)

	if False:
		model = transformers.OwlViTModel.from_pretrained("google/owlvit-base-patch32")
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		inputs = processor(images=image, return_tensors="pt")

		image_features = model.get_image_features(**inputs)

	if False:
		model = transformers.OwlViTTextModel.from_pretrained("google/owlvit-base-patch32")
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")

		inputs = processor(text=[["a photo of a cat", "a photo of a dog"], ["photo of a astranaut"]], return_tensors="pt")

		outputs = model(**inputs)

		last_hidden_state = outputs.last_hidden_state
		pooled_output = outputs.pooler_output  # Pooled (EOS token) states.

	if False:
		model = transformers.OwlViTVisionModel.from_pretrained("google/owlvit-base-patch32")
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		inputs = processor(images=image, return_tensors="pt")

		outputs = model(**inputs)

		last_hidden_state = outputs.last_hidden_state
		pooled_output = outputs.pooler_output  # Pooled CLS states.

	if True:
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch32")
		model = transformers.OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		texts = [["a photo of a cat", "a photo of a dog"]]
		inputs = processor(text=texts, images=image, return_tensors="pt")

		outputs = model(**inputs)

		# Target image sizes (height, width) to rescale box predictions [batch_size, 2].
		target_sizes = torch.Tensor([image.size[::-1]])
		# Convert outputs (bounding boxes and class logits) to final bounding boxes and scores.
		results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

		i = 0  # Retrieve predictions for the first image for the corresponding text queries.
		text = texts[i]
		boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

		for box, score, label in zip(boxes, scores, labels):
			box = [round(i, 2) for i in box.tolist()]
			print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

	if True:
		processor = transformers.AutoProcessor.from_pretrained("google/owlvit-base-patch16")
		model = transformers.OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")  # ~611MB.
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		query_url = "http://images.cocodataset.org/val2017/000000001675.jpg"
		query_image = Image.open(requests.get(query_url, stream=True).raw)
		inputs = processor(images=image, query_images=query_image, return_tensors="pt")

		with torch.no_grad():
			outputs = model.image_guided_detection(**inputs)

		# Target image sizes (height, width) to rescale box predictions [batch_size, 2].
		target_sizes = torch.Tensor([image.size[::-1]])
		# Convert outputs (bounding boxes and class logits) to COCO API.
		results = processor.post_process_image_guided_detection(outputs=outputs, threshold=0.6, nms_threshold=0.3, target_sizes=target_sizes)

		i = 0  # Retrieve predictions for the first image.
		boxes, scores = results[i]["boxes"], results[i]["scores"]
		for box, score in zip(boxes, scores):
			box = [round(i, 2) for i in box.tolist()]
			print(f"Detected similar object with confidence {round(score.item(), 3)} at location {box}")

	if True:
		processor = transformers.OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
		model = transformers.OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")  # ~613MB.

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)
		texts = [["a photo of a cat", "a photo of a dog"]]
		inputs = processor(text=texts, images=image, return_tensors="pt")

		outputs = model(**inputs)

		# Target image sizes (height, width) to rescale box predictions [batch_size, 2].
		target_sizes = torch.Tensor([image.size[::-1]])
		# Convert outputs (bounding boxes and class logits) to COCO API.
		results = processor.post_process(outputs=outputs, target_sizes=target_sizes)

		i = 0  # Retrieve predictions for the first image for the corresponding text queries.
		text = texts[i]
		boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

		score_threshold = 0.1
		for box, score, label in zip(boxes, scores, labels):
			box = [round(i, 2) for i in box.tolist()]
			if score >= score_threshold:
				print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

def main():
	# Open vocabulary object detection.
	owl_vit_test()  # OWL-ViT.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
