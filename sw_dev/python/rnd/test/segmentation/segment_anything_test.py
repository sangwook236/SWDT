#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def facebook_prediction_example():
	import numpy as np
	import torch
	import cv2
	import matplotlib.pyplot as plt
	import segment_anything

	def show_mask(mask, ax, random_color=False):
		if random_color:
			color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
		else:
			color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
		h, w = mask.shape[-2:]
		mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
		ax.imshow(mask_image)
		
	def show_points(coords, labels, ax, marker_size=375):
		pos_points = coords[labels==1]
		neg_points = coords[labels==0]
		ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
		ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
		
	def show_box(box, ax):
		x0, y0 = box[0], box[1]
		w, h = box[2] - box[0], box[3] - box[1]
		ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0,0,0,0), lw=2))

	# Example image.
	image = cv2.imread("./truck.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	plt.axis("on")
	plt.show()

	# Selecting objects with SAM.
	if False:
		sam_checkpoint = "sam_vit_h_4b8939.pth"
		model_type = "vit_h"
	else:
		sam_checkpoint = "sam_vit_l_0b3195.pth"
		model_type = "vit_l"

	device = "cuda" if torch.cuda.is_available() else "cpu"

	sam = segment_anything.sam_model_registry[model_type](checkpoint=sam_checkpoint)
	sam.to(device=device)

	predictor = segment_anything.SamPredictor(sam)
	predictor.set_image(image)

	input_point = np.array([[500, 375]])
	input_label = np.array([1])

	plt.figure(figsize=(10,10))
	plt.imshow(image)
	show_points(input_point, input_label, plt.gca())
	plt.axis("on")
	plt.show()

	masks, scores, logits = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		multimask_output=True,
	)

	print(f"{masks.shape=}.")  # (number_of_masks) x H x W.

	for i, (mask, score) in enumerate(zip(masks, scores)):
		plt.figure(figsize=(10, 10))
		plt.imshow(image)
		show_mask(mask, plt.gca())
		show_points(input_point, input_label, plt.gca())
		plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
		plt.axis("off")
		plt.show()

	# Specifying a specific object with additional points.
	input_point = np.array([[500, 375], [1125, 625]])
	input_label = np.array([1, 1])

	mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask.

	masks, _, _ = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		mask_input=mask_input[None, :, :],
		multimask_output=False,
	)

	print(f"{masks.shape=}.")  # (number_of_masks) x H x W.

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	show_mask(masks, plt.gca())
	show_points(input_point, input_label, plt.gca())
	plt.axis("off")
	plt.show()

	input_point = np.array([[500, 375], [1125, 625]])
	input_label = np.array([1, 0])

	mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask.

	masks, _, _ = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		mask_input=mask_input[None, :, :],
		multimask_output=False,
	)

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	show_mask(masks, plt.gca())
	show_points(input_point, input_label, plt.gca())
	plt.axis("off")
	plt.show() 

	# Specifying a specific object with a box.
	input_box = np.array([425, 600, 700, 875])

	masks, _, _ = predictor.predict(
		point_coords=None,
		point_labels=None,
		box=input_box[None, :],
		multimask_output=False,
	)

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	show_mask(masks[0], plt.gca())
	show_box(input_box, plt.gca())
	plt.axis("off")
	plt.show()

	# Combining points and boxes.
	input_box = np.array([425, 600, 700, 875])
	input_point = np.array([[575, 750]])
	input_label = np.array([0])

	masks, _, _ = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		box=input_box,
		multimask_output=False,
	)

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	show_mask(masks[0], plt.gca())
	show_box(input_box, plt.gca())
	show_points(input_point, input_label, plt.gca())
	plt.axis("off")
	plt.show()

	# Batched prompt inputs.
	input_boxes = torch.tensor([
		[75, 275, 1725, 850],
		[425, 600, 700, 875],
		[1375, 550, 1650, 800],
		[1240, 675, 1400, 750],
	], device=predictor.device)

	transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
	masks, _, _ = predictor.predict_torch(
		point_coords=None,
		point_labels=None,
		boxes=transformed_boxes,
		multimask_output=False,
	)

	print(f"{masks.shape=}.")  # (batch_size) x (num_predicted_masks_per_input) x H x W.

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	for mask in masks:
		show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
	for box in input_boxes:
		show_box(box.cpu().numpy(), plt.gca())
	plt.axis("off")
	plt.show()

	# End-to-end batched inference.
	image1 = image  # truck.jpg from above.
	image1_boxes = torch.tensor([
		[75, 275, 1725, 850],
		[425, 600, 700, 875],
		[1375, 550, 1650, 800],
		[1240, 675, 1400, 750],
	], device=sam.device)

	image2 = cv2.imread("./groceries.jpg")
	image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
	image2_boxes = torch.tensor([
		[450, 170, 520, 350],
		[350, 190, 450, 350],
		[500, 170, 580, 350],
		[580, 170, 640, 350],
	], device=sam.device)

	resize_transform = segment_anything.utils.transforms.ResizeLongestSide(sam.image_encoder.img_size)

	def prepare_image(image, transform, device):
		image = transform.apply_image(image)
		image = torch.as_tensor(image, device=device.device) 
		return image.permute(2, 0, 1).contiguous()

	batched_input = [
		{
			"image": prepare_image(image1, resize_transform, sam),
			"boxes": resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
			"original_size": image1.shape[:2]
		},
		{
			"image": prepare_image(image2, resize_transform, sam),
			"boxes": resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
			"original_size": image2.shape[:2]
		}
	]

	batched_output = sam(batched_input, multimask_output=False)
	print(f"{batched_output[0].keys()=}.")

	fig, ax = plt.subplots(1, 2, figsize=(20, 20))

	ax[0].imshow(image1)
	for mask in batched_output[0]["masks"]:
		show_mask(mask.cpu().numpy(), ax[0], random_color=True)
	for box in image1_boxes:
		show_box(box.cpu().numpy(), ax[0])
	ax[0].axis("off")

	ax[1].imshow(image2)
	for mask in batched_output[1]["masks"]:
		show_mask(mask.cpu().numpy(), ax[1], random_color=True)
	for box in image2_boxes:
		show_box(box.cpu().numpy(), ax[1])
	ax[1].axis("off")

	plt.tight_layout()
	plt.show()

# REF [site] >> https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
def facebook_automatic_mask_generator_example():
	import numpy as np
	import torch
	import cv2
	import matplotlib.pyplot as plt
	import segment_anything

	def show_anns(anns):
		if len(anns) == 0:
			return
		sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
		ax = plt.gca()
		ax.set_autoscale_on(False)

		img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
		img[:,:,3] = 0
		for ann in sorted_anns:
			m = ann["segmentation"]
			color_mask = np.concatenate([np.random.random(3), [0.35]])
			img[m] = color_mask
		ax.imshow(img)

	# Example image.
	image = cv2.imread("./dog.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	plt.figure(figsize=(20,20))
	plt.imshow(image)
	plt.axis("off")
	plt.show()

	# Automatic mask generation.
	if False:
		sam_checkpoint = "sam_vit_h_4b8939.pth"
		model_type = "vit_h"
	else:
		sam_checkpoint = "sam_vit_l_0b3195.pth"
		model_type = "vit_l"

	device = "cuda" if torch.cuda.is_available() else "cpu"

	sam = segment_anything.sam_model_registry[model_type](checkpoint=sam_checkpoint)
	sam.to(device=device)

	mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)

	masks = mask_generator.generate(image)

	print(f"{len(masks)=}.")
	print(f"{masks[0].keys()=}.")
	"""
	segmentation : the mask.
	area : the area of the mask in pixels.
	bbox : the boundary box of the mask in XYWH format.
	predicted_iou : the model's own prediction for the quality of the mask.
	point_coords : the sampled input point that generated this mask.
	stability_score : an additional measure of mask quality.
	crop_box : the crop of the image used to generate this mask in XYWH format.
	"""

	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	show_anns(masks)
	plt.axis("off")
	plt.show()

	# Automatic mask generation options.
	mask_generator_2 = segment_anything.SamAutomaticMaskGenerator(
		model=sam,
		points_per_side=32,
		pred_iou_thresh=0.86,
		stability_score_thresh=0.92,
		crop_n_layers=1,
		crop_n_points_downscale_factor=2,
		min_mask_region_area=100,  # Requires opencv to run post-processing.
	)

	masks2 = mask_generator_2.generate(image)

	print(f"{len(masks2)=}.")

	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	show_anns(masks2)
	plt.axis("off")
	plt.show()

def main():
	facebook_prediction_example()
	#facebook_automatic_mask_generator_example()

	# Hugging Face Segment Anything model.
	#	Refer to ./hugging_face_segmentation_test.py.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
