#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from PIL import Image
import requests

# REF [site] >> https://huggingface.co/docs/transformers/tasks/monocular_depth_estimation
def monocular_depth_estimation_example():
	import transformers
	import matplotlib

	device = "cuda" if torch.cuda.is_available() else "cpu"

	if True:
		checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
		pipe = transformers.pipeline("depth-estimation", model=checkpoint, device=device)

		url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		predictions = pipe(image)
		predictions["depth"]

	if True:
		checkpoint = "Intel/zoedepth-nyu-kitti"

		image_processor = transformers.AutoImageProcessor.from_pretrained(checkpoint)
		model = transformers.AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)

		pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

		with torch.no_grad():
			outputs = model(pixel_values)

		predicted_depth = outputs.predicted_depth.unsqueeze(dim=1)
		height, width = pixel_values.shape[2:]

		height_padding_factor = width_padding_factor = 3
		pad_h = int(np.sqrt(height/2) * height_padding_factor)
		pad_w = int(np.sqrt(width/2) * width_padding_factor)

		if predicted_depth.shape[-2:] != pixel_values.shape[-2:]:
			predicted_depth = torch.nn.functional.interpolate(predicted_depth, size= (height, width), mode="bicubic", align_corners=False)

		if pad_h > 0:
			predicted_depth = predicted_depth[:, :, pad_h:-pad_h, :]
		if pad_w > 0:
			predicted_depth = predicted_depth[:, :, :, pad_w:-pad_w]

		def colorize(value, vmin=None, vmax=None, cmap="gray_r", invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
			"""Converts a depth map to a color image.

			Args:
				value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
				vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
				vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
				cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
				invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
				invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
				background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
				gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
				value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

			Returns:
				numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
			"""
			if isinstance(value, torch.Tensor):
				value = value.detach().cpu().numpy()

			value = value.squeeze()
			if invalid_mask is None:
				invalid_mask = value == invalid_val
			mask = np.logical_not(invalid_mask)

			# Normalize
			vmin = np.percentile(value[mask],2) if vmin is None else vmin
			vmax = np.percentile(value[mask],85) if vmax is None else vmax
			if vmin != vmax:
				value = (value - vmin) / (vmax - vmin)  # vmin..vmax
			else:
				# Avoid 0-division
				value = value * 0.

			# Squeeze last dim if it exists
			# Grey out the invalid values

			value[invalid_mask] = np.nan
			cmapper = matplotlib.colormaps.get_cmap(cmap)
			if value_transform:
				value = value_transform(value)
				#value = value / value.max()
			value = cmapper(value, bytes=True)  # (nxmx4)

			#img = value[:, :, :]
			img = value[...]
			img[invalid_mask] = background_color

			#return img.transpose((2, 0, 1))
			if gamma_corrected:
				# Gamma correction
				img = img / 255
				img = np.power(img, 2.2)
				img = img * 255
				img = img.astype(np.uint8)
			return img

		result = colorize(predicted_depth.cpu().squeeze().numpy())
		Image.fromarray(result)

# REF [site] >> https://huggingface.co/Intel
def dpt_example():
	# Models:
	#	Intel/dpt-large
	#	Intel/dpt-beit-base-384
	#	Intel/dpt-beit-large-384
	#	Intel/dpt-beit-large-512
	#	Intel/dpt-swinv2-tiny-256
	#	Intel/dpt-swinv2-base-384
	#	Intel/dpt-swinv2-large-384
	#	Intel/dpt-hybrid-midas
	#	Intel/dpt-large-ade

	import transformers

	if True:
		# REF [site] >> https://huggingface.co/tasks/depth-estimation

		model_id = "Intel/dpt-large"
		#model_id = "Intel/dpt-beit-base-384"
		#model_id = "Intel/dpt-beit-large-384"
		#model_id = "Intel/dpt-beit-large-512"
		#model_id = "Intel/dpt-swinv2-tiny-256"
		#model_id = "Intel/dpt-swinv2-base-384"
		#model_id = "Intel/dpt-swinv2-large-384"

		pipe = transformers.pipeline(task="depth-estimation", model=model_id)
		result = pipe(images="http://images.cocodataset.org/val2017/000000039769.jpg")
		#result = pipe(images="http://images.cocodataset.org/val2017/000000181816.jpg")
		print(result)
		#print(result[0]["predicted_depth"])  # Tensor
		#print(result[0]["depth"])  # PIL.Image

	if True:
		model_id = "Intel/dpt-large"
		#model_id = "Intel/dpt-beit-base-384"
		#model_id = "Intel/dpt-beit-large-384"
		#model_id = "Intel/dpt-beit-large-512"
		#model_id = "Intel/dpt-swinv2-base-384"
		#model_id = "Intel/dpt-hybrid-midas"

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		processor = transformers.DPTImageProcessor.from_pretrained(model_id)
		model = transformers.DPTForDepthEstimation.from_pretrained(model_id)
		#model = transformers.DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)

		# Prepare image for the model
		inputs = processor(images=image, return_tensors="pt")

		with torch.no_grad():
			outputs = model(**inputs)
			predicted_depth = outputs.predicted_depth

		# Interpolate to original size
		prediction = torch.nn.functional.interpolate(
			predicted_depth.unsqueeze(1),
			size=image.size[::-1],
			mode="bicubic",
			align_corners=False,
		)

		# Visualize the prediction
		output = prediction.squeeze().cpu().numpy()
		formatted = (output * 255 / np.max(output)).astype("uint8")
		depth = Image.fromarray(formatted)

	if True:
		url = "http://images.cocodataset.org/val2017/000000026204.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		feature_extractor = transformers.DPTImageProcessor .from_pretrained("Intel/dpt-large-ade")
		model = transformers.DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

		inputs = feature_extractor(images=image, return_tensors="pt")

		outputs = model(**inputs)
		logits = outputs.logits
		print(logits.shape)
		logits
		prediction = torch.nn.functional.interpolate(
			logits,
			size=image.size[::-1],  # Reverse the size of the original image (width, height)
			mode="bicubic",
			align_corners=False
		)

		# Convert logits to class predictions
		prediction = torch.argmax(prediction, dim=1) + 1

		# Squeeze the prediction tensor to remove dimensions
		prediction = prediction.squeeze()

		# Move the prediction tensor to the CPU and convert it to a numpy array
		prediction = prediction.cpu().numpy()

		# Convert the prediction array to an image
		predicted_seg = Image.fromarray(prediction.squeeze().astype("uint8"))

		# Define the ADE20K palette
		adepallete = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]

		# Apply the color map to the predicted segmentation image
		predicted_seg.putpalette(adepallete)

		# Blend the original image and the predicted segmentation image
		out = Image.blend(image, predicted_seg.convert("RGB"), alpha=0.5)

		print(out)

# REF [site] >> https://huggingface.co/Intel
def zoe_depth_example():
	# Models:
	#	Intel/zoedepth-nyu
	#	Intel/zoedepth-kitti
	#	Intel/zoedepth-nyu-kitti

	import transformers

	model_id = "Intel/zoedepth-nyu"
	#model_id = "Intel/zoedepth-kitti"
	#model_id = "Intel/zoedepth-nyu-kitti"

	# Load pipe
	depth_estimator = transformers.pipeline(task="depth-estimation", model=model_id)

	# Load image
	url = "http://images.cocodataset.org/val2017/000000039769.jpg"
	image = Image.open(requests.get(url, stream=True).raw)

	# Inference
	outputs = depth_estimator(image)
	depth = outputs.depth

# REF [site] >> https://github.com/lpiccinelli-eth/UniDepth
def unidepth_example():
	# Install:
	#	git clone https://github.com/lpiccinelli-eth/UniDepth
	#	cd ${UniDepth_HOME}
	#	pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118
	#		Windows does not support triton. (?)
	#	Install Pillow-SIMD (optional):
	#		pip uninstall pillow
	#		CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

	image_width, image_height, image_depth_channels, image_color_channels = 400, 448, 1, 3

	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0001_TA/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0001_TB/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0002_TA/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0002_TB/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0002_TC/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0003_TA/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0003_TB/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0003_TC/raw"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0004_TA"
	#dir_path = "/work/inno3d/Scanner_Testing_20240911/ST_0005_CA"
	dir_path = "/work/inno3d/240717_upper_full_arch/ResultData_1"
	#dir_path = "/work/inno3d/240717_upper_full_arch/ResultData_2"
	#dir_path = "/work/inno3d/240717_upper_full_arch/ResultData_test"

	if True:
		print(f"Loading scanner data from {dir_path}...")
		start_time = time.time()
		scanner_dat = scanner_utils.load_scanner_data_from_dir(dir_path, image_width, image_height, image_depth_channels, image_color_channels)
		if scanner_dat is None:
			print(f"Failed to load scanner data from {dir_path}.")
			return
		print(f"Scanner data loaded: {time.time() - start_time} secs.")
		depth_images, rgb_images = scanner_dat
		print(f"Depth images: shape = {depth_images.shape}, dtype = {depth_images.dtype}, (min, max) = ({np.min(depth_images)}, {np.max(depth_images)}).")
		print(f"RGB images: shape = {rgb_images.shape}, dtype = {rgb_images.dtype}, (min, max) = ({np.min(rgb_images)}, {np.max(rgb_images)}).")
	else:
		print(f"Loading scanner RGB data from {dir_path}...")
		start_time = time.time()
		rgb_images = scanner_utils.load_scanner_color_data_from_dir(dir_path, image_width, image_height, image_color_channels)
		if rgb_images is None:
			print(f"Failed to load scanner RGB data from {dir_path}.")
			return
		print(f"Scanner RGB data loaded: {time.time() - start_time} secs.")
		print(f"RGB images: shape = {rgb_images.shape}, dtype = {rgb_images.dtype}, (min, max) = ({np.min(rgb_images)}, {np.max(rgb_images)}).")
		depth_images = [None] * len(rgb_images)

	from unidepth.models import UniDepthV1, UniDepthV2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Deivce: {device}.")

	if False:
		#model_name = "lpiccinelli/unidepth-v1-cnvnxtl"
		model_name = "lpiccinelli/unidepth-v1-vitl14"

		model = UniDepthV1.from_pretrained(model_name)
	elif True:
		model_name = "lpiccinelli/unidepth-v2-vits14"
		#model_name = "lpiccinelli/unidepth-v2-vitl14"

		model = UniDepthV2.from_pretrained(model_name)
	elif False:
		version = "v2"
		backbone = "vitl14"

		model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)
	else:
		raise ValueError("Model not selected.")
	model.resolution_level = 10  # TODO [check] >>
	model = model.to(device)

	for depth_img, rgb_img in zip(depth_images, rgb_images):
	#for depth_img, rgb_img in zip(depth_images[:10], rgb_images[:10]):
		# Load the RGB image and the normalization will be taken care of by the model
		rgb = torch.from_numpy(rgb_img).permute(2, 0, 1)  # C, H, W

		predictions = model.infer(rgb)

		# Metric depth estimation
		depth = predictions["depth"]
		print(f"Depth: shape = {depth.shape}, dtype = {depth.dtype}, (min, max) = ({torch.min(depth)}, {torch.max(depth)}).")

		# Point cloud in camera coordinate
		pcloud = predictions["points"]
		print(f"Point cloud: shape = {pcloud.shape}, dtype = {pcloud.dtype}, (min, max) = ({torch.min(pcloud)}, {torch.max(pcloud)}).")

		# Intrinsics prediction
		intrinsics = predictions["intrinsics"]
		print(f"Intrinsics: shape = {intrinsics.shape}, dtype = {intrinsics.dtype}, (min, max) = ({torch.min(intrinsics)}, {torch.max(intrinsics)}).")

		# Visualize the prediction
		predictions = depth.squeeze().cpu().numpy()
		#predictions_scaled = (predictions * 255 / np.max(predictions)).astype("uint8")

		if depth_img is None:
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), tight_layout=True)
			ax[0].imshow(rgb_img)
			ax[0].axis("off")
			ax[0].set_title("RGB Image")
			#im = ax[1].imshow(predictions_scaled)
			im = ax[1].imshow(np.float64(predictions))
			fig.colorbar(im, ax=ax[1])
			ax[1].axis("off")
			ax[1].set_title("Depth Prediction")
		else:
			fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), tight_layout=True)
			ax[0].imshow(rgb_img)
			ax[0].axis("off")
			ax[0].set_title("RGB Image")
			im = ax[1].imshow(depth_img)
			fig.colorbar(im, ax=ax[1])
			ax[1].axis("off")
			ax[1].set_title("Depth Image")
			#im = ax[2].imshow(predictions_scaled)
			im = ax[2].imshow(np.float64(predictions))
			fig.colorbar(im, ax=ax[2])
			ax[2].axis("off")
			ax[2].set_title("Depth Prediction")

		plt.show()

# REF [site] >> https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
def depth_anything_example():
	import transformers

	if True:
		# Load pipe
		pipe = transformers.pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

		# Load image
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		# Inference
		depth = pipe(image)["depth"]

	if True:
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
		model = transformers.AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

		# Prepare image for the model
		inputs = image_processor(images=image, return_tensors="pt")

		with torch.no_grad():
			outputs = model(**inputs)
			predicted_depth = outputs.predicted_depth

		# Interpolate to original size
		prediction = torch.nn.functional.interpolate(
			predicted_depth.unsqueeze(1),
			size=image.size[::-1],
			mode="bicubic",
			align_corners=False,
		)

		# Visualize the prediction
		output = prediction.squeeze().cpu().numpy()
		formatted = (output * 255 / np.max(output)).astype("uint8")
		depth = Image.fromarray(formatted)

	if False:
		# Initializing a DepthAnything small style configuration
		configuration = transformers.DepthAnythingConfig()

		# Initializing a model from the DepthAnything small style configuration
		model = transformers.DepthAnythingForDepthEstimation(configuration)

		# Accessing the model configuration
		configuration = model.config

# REF [site] >>
#	https://huggingface.co/depth-anything
#	https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
def depth_anything_v2_example():
	# Models:
	#	depth-anything/Depth-Anything-V2-Small
	#	depth-anything/Depth-Anything-V2-Small-hf
	#	depth-anything/Depth-Anything-V2-Base
	#	depth-anything/Depth-Anything-V2-Base-hf
	#	depth-anything/Depth-Anything-V2-Large
	#	depth-anything/Depth-Anything-V2-Large-hf
	#
	#	depth-anything/Depth-Anything-V2-Metric-Hypersim-Small
	#	depth-anything/Depth-Anything-V2-Metric-Hypersim-Base
	#	depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
	#	depth-anything/Depth-Anything-V2-Metric-VKITTI-Small
	#	depth-anything/Depth-Anything-V2-Metric-VKITTI-Base
	#	depth-anything/Depth-Anything-V2-Metric-VKITTI-Large
	#
	#	depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
	#	depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf
	#	depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf
	#	depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
	#	depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
	#	depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf

	import transformers

	if True:
		# Install:
		#	git clone https://huggingface.co/spaces/depth-anything/Depth-Anything-V2
		#	cd Depth-Anything-V2
		#	pip install -r requirements.txt

		import cv2
		from depth_anything_v2.dpt import DepthAnythingV2

		if False:
			model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
			model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vits.pth", map_location="cpu"))
		elif True:
			model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
			model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location="cpu"))
		elif False:
			model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
			model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitl.pth", map_location="cpu"))
		model.eval()

		raw_img = cv2.imread("/path/to/image")
		depth = model.infer_image(raw_img)  # HxW raw depth map

	if True:
		import cv2
		from depth_anything_v2.dpt import DepthAnythingV2

		model_configs = {
			"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
			"vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
			"vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
		}

		encoder = "vitl"  # or 'vits', 'vitb'
		dataset = "hypersim"  # 'hypersim' for indoor model, 'vkitti' for outdoor model
		max_depth = 20  # 20 for indoor model, 80 for outdoor model

		model = DepthAnythingV2(**{**model_configs[encoder], "max_depth": max_depth})
		model.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth", map_location="cpu"))
		model.eval()

		raw_img = cv2.imread("/path/to/image")
		depth = model.infer_image(raw_img)  # HxW depth map in meters in numpy

	if True:
		#model_id = "depth-anything/Depth-Anything-V2-Small-hf"
		model_id = "depth-anything/Depth-Anything-V2-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Large-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

		# Load pipe
		pipe = transformers.pipeline(task="depth-estimation", model=model_id)

		# Load image
		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		# Inference
		depth = pipe(image)["depth"]

	if True:
		#model_id = "depth-anything/Depth-Anything-V2-Small-hf"
		model_id = "depth-anything/Depth-Anything-V2-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Large-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
		#model_id = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

		url = "http://images.cocodataset.org/val2017/000000039769.jpg"
		image = Image.open(requests.get(url, stream=True).raw)

		image_processor = transformers.AutoImageProcessor.from_pretrained(model_id)
		model = transformers.AutoModelForDepthEstimation.from_pretrained(model_id)

		# Prepare image for the model
		inputs = image_processor(images=image, return_tensors="pt")

		with torch.no_grad():
			outputs = model(**inputs)
			predicted_depth = outputs.predicted_depth

		# Interpolate to original size
		prediction = torch.nn.functional.interpolate(
			predicted_depth.unsqueeze(1),
			size=image.size[::-1],
			mode="bicubic",
			align_corners=False,
		)

		# Visualize the prediction
		output = prediction.squeeze().cpu().numpy()
		formatted = (output * 255 / np.max(output)).astype("uint8")
		depth = Image.fromarray(formatted)

	if False:
		# Initializing a DepthAnything small style configuration
		configuration = transformers.DepthAnythingConfig()

		# Initializing a model from the DepthAnything small style configuration
		model = transformers.DepthAnythingForDepthEstimation(configuration)

		# Accessing the model configuration
		configuration = model.config

def main():
	# https://huggingface.co/docs/datasets/en/depth_estimation
	# Refer to ldm3d_example() in # ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/diffusion_model_test.py

	#monocular_depth_estimation_example()

	# Dense Prediction Transformer (DPT)
	#dpt_example()  # ${SWDT_PYTHON_HOME}/rnd/test/language_processing/hugging_face_transformers_test.py
	#dpt_example()

	# MiDaS
	#	https://pytorch.org/hub/intelisl_midas_v2/
	#	https://github.com/isl-org/MiDaS

	# ZoeDepth
	#zoe_depth_example()

	# Marigold
	#	https://github.com/prs-eth/Marigold
	#	https://huggingface.co/prs-eth

	# Unidepth
	unidepth_example()

	# Depth Anything
	#depth_anything_example()
	#depth_anything_v2_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
