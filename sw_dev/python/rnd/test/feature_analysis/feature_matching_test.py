#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://huggingface.co/docs/transformers/en/model_doc/superpoint
def superpoint_example():
	import transformers
	import torch
	from PIL import Image
	import requests
	import matplotlib.pyplot as plt

	if True:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
		model = transformers.SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

		inputs = processor(image, return_tensors="pt")
		outputs = model(**inputs)

	if True:
		url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
		url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
		image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

		images = [image_1, image_2]

		processor = transformers.AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
		model = transformers.SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

		inputs = processor(images, return_tensors="pt")
		outputs = model(**inputs)
		image_sizes = [(image.height, image.width) for image in images]
		outputs = processor.post_process_keypoint_detection(outputs, image_sizes)

		for output in outputs:
			for keypoints, scores, descriptors in zip(output["keypoints"], output["scores"], output["descriptors"]):
				print(f"Keypoints: {keypoints}")
				print(f"Scores: {scores}")
				print(f"Descriptors: {descriptors}")

		plt.axis("off")
		plt.imshow(image_1)
		plt.scatter(
			outputs[0]["keypoints"][:, 0],
			outputs[0]["keypoints"][:, 1],
			c=outputs[0]["scores"] * 100,
			s=outputs[0]["scores"] * 50,
			alpha=0.8
		)
		plt.savefig(f"output_image.png")

	if False:
		# Initializing a SuperPoint superpoint style configuration
		configuration = transformers.SuperPointConfig()
		# Initializing a model from the superpoint style configuration
		model = transformers.SuperPointForKeypointDetection(configuration)
		# Accessing the model configuration
		configuration = model.config

# REF [site] >> https://huggingface.co/docs/transformers/en/model_doc/superglue
def superglue_example():
	import transformers
	import torch
	from PIL import Image
	import requests
	import matplotlib.pyplot as plt
	import numpy as np

	if True:
		url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
		image1 = Image.open(requests.get(url_image1, stream=True).raw)
		url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
		image2 = Image.open(requests.get(url_image2, stream=True).raw)

		images = [image1, image2]

		processor = transformers.AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
		model = transformers.AutoModel.AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

		inputs = processor(images, return_tensors="pt")
		with torch.no_grad():
			outputs = model(**inputs)

		image_sizes = [[(image.height, image.width) for image in images]]
		outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
		for i, output in enumerate(outputs):
			print("For the image pair", i)
			for keypoint0, keypoint1, matching_score in zip(
					output["keypoints0"], output["keypoints1"], output["matching_scores"]
			):
				print(
					f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
				)

		# Create side by side image
		merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
		merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
		merged_image[: image2.height, image1.width :] = np.array(image2) / 255.0
		plt.imshow(merged_image)
		plt.axis("off")

		# Retrieve the keypoints and matches
		output = outputs[0]
		keypoints0 = output["keypoints0"]
		keypoints1 = output["keypoints1"]
		matching_scores = output["matching_scores"]
		keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
		keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

		# Plot the matches
		for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
				keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
		):
			plt.plot(
				[keypoint0_x, keypoint1_x + image1.width],
				[keypoint0_y, keypoint1_y],
				color=plt.get_cmap("RdYlGn")(matching_score.item()),
				alpha=0.9,
				linewidth=0.5,
			)
			plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
			plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)

		# Save the plot
		plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
		plt.close()

	if False:
		# Initializing a SuperGlue superglue style configuration
		configuration = transformers.SuperGlueConfig()

		# Initializing a model from the superglue style configuration
		model = transformers.SuperGlueModel(configuration)

		# Accessing the model configuration
		configuration = model.config

def main():
	# Image Matching WebUI (IMCUI)
	#	https://huggingface.co/spaces/Realcat/image-matching-webui
	#	https://github.com/Vincentqyw/image-matching-webui

	superpoint_example()
	superglue_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
