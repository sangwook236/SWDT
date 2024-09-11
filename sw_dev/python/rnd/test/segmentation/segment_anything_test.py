#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def facebook_sam_predictor_example():
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
def facebook_sam_automatic_mask_generator_example():
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

# REF [site] >> https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb
def facebook_sam2_image_predictor_example():
	import os
	# If using Apple MPS, fall back to CPU for unsupported ops
	os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
	import numpy as np
	import torch
	import matplotlib.pyplot as plt
	from PIL import Image

	# Set-up

	# Select the device for computation
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	print(f"using device: {device}")

	if device.type == "cuda":
		# Use bfloat16 for the entire notebook
		torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
		# Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
		if torch.cuda.get_device_properties(0).major >= 8:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
	elif device.type == "mps":
		print(
			"\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
			"give numerically different outputs and sometimes degraded performance on MPS. "
			"See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
		)

	np.random.seed(3)

	def show_mask(mask, ax, random_color=False, borders = True):
		if random_color:
			color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
		else:
			color = np.array([30/255, 144/255, 255/255, 0.6])
		h, w = mask.shape[-2:]
		mask = mask.astype(np.uint8)
		mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
		if borders:
			import cv2
			contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
			# Try to smooth contours
			contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
			mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
		ax.imshow(mask_image)

	def show_points(coords, labels, ax, marker_size=375):
		pos_points = coords[labels==1]
		neg_points = coords[labels==0]
		ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
		ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)   

	def show_box(box, ax):
		x0, y0 = box[0], box[1]
		w, h = box[2] - box[0], box[3] - box[1]
		ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))    

	def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
		for i, (mask, score) in enumerate(zip(masks, scores)):
			plt.figure(figsize=(10, 10))
			plt.imshow(image)
			show_mask(mask, plt.gca(), borders=borders)
			if point_coords is not None:
				assert input_labels is not None
				show_points(point_coords, input_labels, plt.gca())
			if box_coords is not None:
				# boxes
				show_box(box_coords, plt.gca())
			if len(scores) > 1:
				plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
			plt.axis("off")
			plt.show()

	# Example image
	image = Image.open("./images/truck.jpg")
	image = np.array(image.convert("RGB"))

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	plt.axis("on")
	plt.show()

	#-----
	# Selecting objects with SAM 2
	from sam2.build_sam import build_sam2
	from sam2.sam2_image_predictor import SAM2ImagePredictor

	sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
	model_cfg = "sam2_hiera_l.yaml"

	sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

	predictor = SAM2ImagePredictor(sam2_model)

	predictor.set_image(image)

	input_point = np.array([[500, 375]])
	input_label = np.array([1])

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	show_points(input_point, input_label, plt.gca())
	plt.axis("on")
	plt.show()

	print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

	masks, scores, logits = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		multimask_output=True,
	)
	sorted_ind = np.argsort(scores)[::-1]
	masks = masks[sorted_ind]
	scores = scores[sorted_ind]
	logits = logits[sorted_ind]

	print(masks.shape)  # (number_of_masks) x H x W

	show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

	#-----
	# Specifying a specific object with additional points
	input_point = np.array([[500, 375], [1125, 625]])
	input_label = np.array([1, 1])

	mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

	masks, scores, _ = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		mask_input=mask_input[None, :, :],
		multimask_output=False,
	)

	print(masks.shape)

	show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

	input_point = np.array([[500, 375], [1125, 625]])
	input_label = np.array([1, 0])

	mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

	masks, scores, _ = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		mask_input=mask_input[None, :, :],
		multimask_output=False,
	)

	show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

	#-----
	# Specifying a specific object with a box
	input_box = np.array([425, 600, 700, 875])

	masks, scores, _ = predictor.predict(
		point_coords=None,
		point_labels=None,
		box=input_box[None, :],
		multimask_output=False,
	)

	show_masks(image, masks, scores, box_coords=input_box)

	#-----
	# Combining points and boxes
	input_box = np.array([425, 600, 700, 875])
	input_point = np.array([[575, 750]])
	input_label = np.array([0])

	masks, scores, logits = predictor.predict(
		point_coords=input_point,
		point_labels=input_label,
		box=input_box,
		multimask_output=False,
	)

	#-----
	# Batched prompt inputs
	input_boxes = np.array([
		[75, 275, 1725, 850],
		[425, 600, 700, 875],
		[1375, 550, 1650, 800],
		[1240, 675, 1400, 750],
	])

	masks, scores, _ = predictor.predict(
		point_coords=None,
		point_labels=None,
		box=input_boxes,
		multimask_output=False,
	)

	print(masks.shape)  # (batch_size) x (num_predicted_masks_per_input) x H x W

	plt.figure(figsize=(10, 10))
	plt.imshow(image)
	for mask in masks:
		show_mask(mask.squeeze(0), plt.gca(), random_color=True)
	for box in input_boxes:
		show_box(box, plt.gca())
	plt.axis("off")
	plt.show()

	#-----
	# End-to-end batched inference
	image1 = image  # truck.jpg from above
	image1_boxes = np.array([
		[75, 275, 1725, 850],
		[425, 600, 700, 875],
		[1375, 550, 1650, 800],
		[1240, 675, 1400, 750],
	])

	image2 = Image.open("./images/groceries.jpg")
	image2 = np.array(image2.convert("RGB"))
	image2_boxes = np.array([
		[450, 170, 520, 350],
		[350, 190, 450, 350],
		[500, 170, 580, 350],
		[580, 170, 640, 350],
	])

	img_batch = [image1, image2]
	boxes_batch = [image1_boxes, image2_boxes]

	predictor.set_image_batch(img_batch)

	masks_batch, scores_batch, _ = predictor.predict_batch(
		None,
		None, 
		box_batch=boxes_batch, 
		multimask_output=False
	)

	for image, boxes, masks in zip(img_batch, boxes_batch, masks_batch):
		plt.figure(figsize=(10, 10))
		plt.imshow(image)   
		for mask in masks:
			show_mask(mask.squeeze(0), plt.gca(), random_color=True)
		for box in boxes:
			show_box(box, plt.gca())

	image1 = image  # truck.jpg from above
	image1_pts = np.array([
		[[500, 375]],
		[[650, 750]]
		]) # Bx1x2 where B corresponds to number of objects 
	image1_labels = np.array([[1], [1]])

	image2_pts = np.array([
		[[400, 300]],
		[[630, 300]],
	])
	image2_labels = np.array([[1], [1]])

	pts_batch = [image1_pts, image2_pts]
	labels_batch = [image1_labels, image2_labels]

	masks_batch, scores_batch, _ = predictor.predict_batch(pts_batch, labels_batch, box_batch=None, multimask_output=True)

	# Select the best single mask per object
	best_masks = []
	for masks, scores in zip(masks_batch,scores_batch):
		best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])

	for image, points, labels, masks in zip(img_batch, pts_batch, labels_batch, best_masks):
		plt.figure(figsize=(10, 10))
		plt.imshow(image)   
		for mask in masks:
			show_mask(mask, plt.gca(), random_color=True)
		show_points(points, labels, plt.gca())


# REF [site] >> https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb
def facebook_sam2_video_predictor_example():
	import os
	# if using Apple MPS, fall back to CPU for unsupported ops
	os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
	import numpy as np
	import torch
	import matplotlib.pyplot as plt
	from PIL import Image

	# Set-up

	# Select the device for computation
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	print(f"using device: {device}")

	if device.type == "cuda":
		# Use bfloat16 for the entire notebook
		torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
		# Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
		if torch.cuda.get_device_properties(0).major >= 8:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
	elif device.type == "mps":
		print(
			"\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
			"give numerically different outputs and sometimes degraded performance on MPS. "
			"See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
		)

	# Loading the SAM 2 video predictor
	from sam2.build_sam import build_sam2_video_predictor

	sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
	model_cfg = "sam2_hiera_l.yaml"

	predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

	def show_mask(mask, ax, obj_id=None, random_color=False):
		if random_color:
			color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
		else:
			cmap = plt.get_cmap("tab10")
			cmap_idx = 0 if obj_id is None else obj_id
			color = np.array([*cmap(cmap_idx)[:3], 0.6])
		h, w = mask.shape[-2:]
		mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
		ax.imshow(mask_image)

	def show_points(coords, labels, ax, marker_size=200):
		pos_points = coords[labels==1]
		neg_points = coords[labels==0]
		ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
		ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

	def show_box(box, ax):
		x0, y0 = box[0], box[1]
		w, h = box[2] - box[0], box[3] - box[1]
		ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

	# Select an example video
	# 'video_dir' a directory of JPEG frames with filenames like `<frame_index>.jpg`
	video_dir = "./videos/bedroom"

	# Scan all the JPEG frame names in this directory
	frame_names = [
		p for p in os.listdir(video_dir)
		if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
	]
	frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

	# Take a look the first video frame
	frame_idx = 0
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

	# Initialize the inference state
	inference_state = predictor.init_state(video_path=video_dir)

	#-----
	# Example 1: Segment & track one object
	predictor.reset_state(inference_state)

	# Step 1: Add a first click on a frame
	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 1  # Give a unique id to each object we interact with (it can be any integers)

	# Let's add a positive click at (x, y) = (210, 350) to get started
	points = np.array([[210, 350]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1], np.int32)
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

	# Step 2: Add a second click to refine the prediction
	ann_frame_idx = 0  # the frame index we interact with
	ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

	# Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
	# sending all clicks (and their labels) to `add_new_points_or_box`
	points = np.array([[210, 350], [250, 220]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1, 1], np.int32)
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

	# Step 3: Propagate the prompts to get the masklet across the video
	# Run propagation throughout the video and collect the results in a dict
	video_segments = {}  # video_segments contains the per-frame segmentation results
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# Render the segmentation results every few frames
	vis_frame_stride = 30
	plt.close("all")
	for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
		plt.figure(figsize=(6, 4))
		plt.title(f"frame {out_frame_idx}")
		plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
		for out_obj_id, out_mask in video_segments[out_frame_idx].items():
			show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

	# Step 4: Add new prompts to further refine the masklet
	ann_frame_idx = 150  # Further refine some details on this frame
	ann_obj_id = 1  # Give a unique id to the object we interact with (it can be any integers)

	# Show the segment before further refinement
	plt.figure(figsize=(12, 8))
	plt.title(f"frame {ann_frame_idx} -- before refinement")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)

	# Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
	points = np.array([[82, 415]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([0], np.int32)
	_, _, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the segment after the further refinement
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx} -- after refinement")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)

	# Step 5: Propagate the prompts (again) to get the masklet across the video
	# Run propagation throughout the video and collect the results in a dict
	video_segments = {}  # video_segments contains the per-frame segmentation results
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# Render the segmentation results every few frames
	vis_frame_stride = 30
	plt.close("all")
	for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
		plt.figure(figsize=(6, 4))
		plt.title(f"frame {out_frame_idx}")
		plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
		for out_obj_id, out_mask in video_segments[out_frame_idx].items():
			show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

	#-----
	# Example 2: Segment an object using box prompt
	predictor.reset_state(inference_state)

	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 4  # Give a unique id to each object we interact with (it can be any integers)

	# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
	box = np.array([300, 0, 500, 400], dtype=np.float32)
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		box=box,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_box(box, plt.gca())
	show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 4  # Give a unique id to each object we interact with (it can be any integers)

	# Let's add a positive click at (x, y) = (460, 60) to refine the mask
	points = np.array([[460, 60]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1], np.int32)
	# Note that we also need to send the original box input along with
	# the new refinement click together into `add_new_points_or_box`
	box = np.array([300, 0, 500, 400], dtype=np.float32)
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
		box=box,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_box(box, plt.gca())
	show_points(points, labels, plt.gca())
	show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

	# Run propagation throughout the video and collect the results in a dict
	video_segments = {}  # video_segments contains the per-frame segmentation results
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# Render the segmentation results every few frames
	vis_frame_stride = 30
	plt.close("all")
	for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
		plt.figure(figsize=(6, 4))
		plt.title(f"frame {out_frame_idx}")
		plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
		for out_obj_id, out_mask in video_segments[out_frame_idx].items():
			show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

	#-----
	# Example 3: Segment multiple objects simultaneously
	predictor.reset_state(inference_state)

	# Step 1: Add two objects on a frame
	prompts = {}  # Hold all the clicks we add for visualization

	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 2  # Give a unique id to each object we interact with (it can be any integers)

	# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
	points = np.array([[200, 300]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1], np.int32)
	prompts[ann_obj_id] = points, labels
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	for i, out_obj_id in enumerate(out_obj_ids):
		show_points(*prompts[out_obj_id], plt.gca())
		show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

	# Add the first object
	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 2  # Give a unique id to each object we interact with (it can be any integers)

	# Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
	# sending all clicks (and their labels) to `add_new_points_or_box`
	points = np.array([[200, 300], [275, 175]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1, 0], np.int32)
	prompts[ann_obj_id] = points, labels
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the results on the current (interacted) frame
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	for i, out_obj_id in enumerate(out_obj_ids):
		show_points(*prompts[out_obj_id], plt.gca())
		show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

	ann_frame_idx = 0  # The frame index we interact with
	ann_obj_id = 3  # Give a unique id to each object we interact with (it can be any integers)

	# Let's now move on to the second object we want to track (giving it object id `3`)
	# with a positive click at (x, y) = (400, 150)
	points = np.array([[400, 150]], dtype=np.float32)
	# For labels, `1` means positive click and `0` means negative click
	labels = np.array([1], np.int32)
	prompts[ann_obj_id] = points, labels

	# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
	_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
		inference_state=inference_state,
		frame_idx=ann_frame_idx,
		obj_id=ann_obj_id,
		points=points,
		labels=labels,
	)

	# Show the results on the current (interacted) frame on all objects
	plt.figure(figsize=(9, 6))
	plt.title(f"frame {ann_frame_idx}")
	plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
	show_points(points, labels, plt.gca())
	for i, out_obj_id in enumerate(out_obj_ids):
		show_points(*prompts[out_obj_id], plt.gca())
		show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

	# Step 2: Propagate the prompts to get masklets across the video
	# Run propagation throughout the video and collect the results in a dict
	video_segments = {}  # video_segments contains the per-frame segmentation results
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# Render the segmentation results every few frames
	vis_frame_stride = 30
	plt.close("all")
	for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
		plt.figure(figsize=(6, 4))
		plt.title(f"frame {out_frame_idx}")
		plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
		for out_obj_id, out_mask in video_segments[out_frame_idx].items():
			show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

# REF [site] >> https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb
def facebook_sam2_automatic_mask_generator_example():
	import os
	# If using Apple MPS, fall back to CPU for unsupported ops
	os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
	import numpy as np
	import torch
	import matplotlib.pyplot as plt
	from PIL import Image

	# Set-up

	# Select the device for computation
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")
	print(f"using device: {device}")

	if device.type == "cuda":
		# Use bfloat16 for the entire notebook
		torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
		# Turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
		if torch.cuda.get_device_properties(0).major >= 8:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
	elif device.type == "mps":
		print(
			"\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
			"give numerically different outputs and sometimes degraded performance on MPS. "
			"See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
		)

	np.random.seed(3)

	def show_anns(anns, borders=True):
		if len(anns) == 0:
			return
		sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
		ax = plt.gca()
		ax.set_autoscale_on(False)

		img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
		img[:, :, 3] = 0
		for ann in sorted_anns:
			m = ann["segmentation"]
			color_mask = np.concatenate([np.random.random(3), [0.5]])
			img[m] = color_mask 
			if borders:
				import cv2
				contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
				# Try to smooth contours
				contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
				cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

		ax.imshow(img)

	# Example image
	image = Image.open("./images/cars.jpg")
	image = np.array(image.convert("RGB"))

	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	plt.axis("off")
	plt.show()

	#-----
	# Automatic mask generation
	from sam2.build_sam import build_sam2
	from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

	sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
	model_cfg = "sam2_hiera_l.yaml"

	sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

	mask_generator = SAM2AutomaticMaskGenerator(sam2)

	masks = mask_generator.generate(image)
	# Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
	#	segmentation: the mask
	#	area: the area of the mask in pixels
	#	bbox: the boundary box of the mask in XYWH format
	#	predicted_iou: the model's own prediction for the quality of the mask
	#	point_coords: the sampled input point that generated this mask
	#	stability_score: an additional measure of mask quality
	#	crop_box: the crop of the image used to generate this mask in XYWH format

	print(len(masks))
	print(masks[0].keys())

	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	show_anns(masks)
	plt.axis("off")
	plt.show()

	#-----
	# Automatic mask generation options
	mask_generator_2 = SAM2AutomaticMaskGenerator(
		model=sam2,
		points_per_side=64,
		points_per_batch=128,
		pred_iou_thresh=0.7,
		stability_score_thresh=0.92,
		stability_score_offset=0.7,
		crop_n_layers=1,
		box_nms_thresh=0.7,
		crop_n_points_downscale_factor=2,
		min_mask_region_area=25.0,
		use_m2m=True,
	)

	masks2 = mask_generator_2.generate(image)

	plt.figure(figsize=(20, 20))
	plt.imshow(image)
	show_anns(masks2)
	plt.axis("off")
	plt.show() 

def main():
	# SAM
	facebook_sam_predictor_example()
	#facebook_sam_automatic_mask_generator_example()

	# SAM 2
	facebook_sam2_image_predictor_example()
	facebook_sam2_video_predictor_example
	#facebook_sam2_automatic_mask_generator_example()

	# Hugging Face Segment Anything model.
	#	Refer to ./hugging_face_segmentation_test.py.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
