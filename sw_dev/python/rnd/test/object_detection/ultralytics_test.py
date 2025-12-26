#!/usr/bin/env python
# -*- coding: UTF-8 -*-

def benchmark():
	from ultralytics.utils.benchmarks import benchmark

	# Benchmark on GPU
	benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)

# REF [site] >> https://docs.ultralytics.com/modes/export/
def export_models_test():
	from ultralytics import YOLO

	# Load a pretrained YOLOv8 model (e.g., yolov8n for nano)
	model = YOLO("yolo11n.pt")  # Load an official model
	#model = YOLO("path/to/best.pt")  # Load a custom-trained model

	# Export the model
	#model.export(format="torchscript")
	#model.export(format="onnx")
	model.export(format="onnx", opset=12, imgsz=640)
	#model.export(format="onnx", opset=12, imgsz=[640, 640])
	#model.export(format="onnx", dynamic=True)  # Dynamic input size
	#model.export(format="engine")  # TensorRT
	#model.export(format="engine", int8=True)  # INT8 quantization

def yolov5_detection_example():
	from ultralytics import YOLO

	# Load a model
	if False:
		model = YOLO("yolov5nu.yaml")  # Build a new model from scratch
	elif True:
		model = YOLO("yolov5nu.pt")  # Load a pretrained model (recommended for training)
	else:
		model = YOLO("path/to/best.pt")  # Load a custom model
	#print("Model info:")
	#print(model.info())

	# Train the model
	#model.train(data="coco128.yaml", epochs=3)  # ultralytics.utils.metrics.DetMetrics
	model.train(data="coco128.yaml", epochs=100, imgsz=640)  # ultralytics.utils.metrics.DetMetrics
	#model.train(data="coco128.yaml", epochs=100, resume=True)  # ultralytics.utils.metrics.DetMetrics

	# Validate the model
	metrics = model.val()  # No arguments needed, dataset and settings remembered. ultralytics.utils.metrics.DetMetrics
	#	metrics.box.map  # mAP50-95
	#	metrics.box.map50  # mAP50
	#	metrics.box.map75  # mAP75
	#	metrics.box.maps  # A list contains mAP50-95 of each category

	# Predict with the model
	results = model("https://ultralytics.com/images/bus.jpg")  # A list of ultralytics.engine.results.Results
	
	# Export the model
	success = model.export(format="onnx")

# REF [site] >>
#	https://github.com/ultralytics/ultralytics
#	https://docs.ultralytics.com/tasks/detection/
def yolov8_detection_example():
	from ultralytics import YOLO

	# Load a model
	if False:
		model = YOLO("yolov8n.yaml")  # Build a new model from scratch
	elif True:
		model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training)
	else:
		model = YOLO("path/to/best.pt")  # Load a custom model
	#print("Model info:")
	#print(model.info())

	# Train the model
	#model.train(data="coco128.yaml", epochs=3)  # ultralytics.utils.metrics.DetMetrics
	model.train(data="coco128.yaml", epochs=100, imgsz=640)  # ultralytics.utils.metrics.DetMetrics

	# Validate the model
	metrics = model.val()  # No arguments needed, dataset and settings remembered. ultralytics.utils.metrics.DetMetrics
	#	metrics.box.map  # mAP50-95
	#	metrics.box.map50  # mAP50
	#	metrics.box.map75  # mAP75
	#	metrics.box.maps  # A list contains mAP50-95 of each category

	# Predict with the model
	results = model("https://ultralytics.com/images/bus.jpg")  # A list of ultralytics.engine.results.Results
	
	# Export the model
	success = model.export(format="onnx")

# REF [site] >> https://docs.ultralytics.com/tasks/segmentation/
def yolov8_instance_segmentation_example():
	from ultralytics import YOLO

	# Load a model
	if False:
		model = YOLO("yolov8n-seg.yaml")  # Build a new model from scratch
	elif True:
		model = YOLO("yolov8n-seg.pt")  # Load a pretrained model (recommended for training)
	else:
		model = YOLO("path/to/best.pt")  # Load a custom model
	#print("Model info:")
	#print(model.info())

	# Train the model
	metrics = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)  # ultralytics.utils.metrics.SegmentationMetrics

	# Validate the model
	metrics = model.val()  # No arguments needed, dataset and settings remembered. ultralytics.utils.metrics.SegmentationMetrics

	# ultralytics.utils.metrics.SegmentMetrics
	#	https://docs.ultralytics.com/reference/utils/metrics/
	#
	#	names: dict[int, str]	Dictionary of class names
	#	box: Metric				An instance of the Metric class for storing detection results
	#	seg: Metric				An instance of the Metric class to calculate mask segmentation metrics
	#	speed: dict[str, float]	A dictionary for storing execution times of different parts of the detection process
	#	task: str				The task type, set to 'segment'
	#	stats: dict[str, list]	A dictionary containing lists for true positives, confidence scores, predicted classes, target classes, and target images
	#	nt_per_class: int		Number of targets per class
	#	nt_per_image: int		Number of targets per image
	#
	#	keys			A list of keys for accessing metrics. ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
	#	maps			mAP scores for object detection and semantic segmentation models
	#	fitness			Fitness score for both segmentation and bounding box models
	#	curves			A list of curves for accessing specific metrics curves. ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)', 'Precision-Recall(M)', 'F1-Confidence(M)', 'Precision-Confidence(M)', 'Recall-Confidence(M)']
	#	curves_results	A list of computed performance metrics and statistics
	#	class_result	Classification results for a specified class index
	#	mean_results	Mean metrics for bounding box and segmentation results
	#	process			The detection and segmentation metrics over the given set of predictions
	#	summary			A summarized representation of per-class segmentation metrics as a list of dictionaries
	#
	# ultralytics.utils.metrics.Metric
	#	p: list					Precision for each class. Shape: (nc,)
	#	r: list					Recall for each class. Shape: (nc,)
	#	f1: list				F1 score for each class. Shape: (nc,)
	#	all_ap: list			AP scores for all classes and all IoU thresholds. Shape: (nc, 10)
	#	ap_class_index: list	Index of class for each AP score. Shape: (nc,)
	#	nc: int					Number of classes
	#
	#	ap50			Return the Average Precision (AP) at an IoU threshold of 0.5 for all classes
	#	ap				Return the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes
	#	mp				Return the Mean Precision of all classes
	#	mr				Return the Mean Recall of all classes
	#	map50			Return the mean Average Precision (mAP) at an IoU threshold of 0.5 (mAP50)
	#	map75			Return the mean Average Precision (mAP) at an IoU threshold of 0.75 (mAP75)
	#	map				Return the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05
	#	maps			Return mAP of each class
	#	curves			Return a list of curves for accessing specific metrics curves
	#	curves_results	Return a list of curves for accessing specific metrics curves
	#	class_result	Return class-aware result, p[i], r[i], ap50[i], ap[i]
	#	fitness			Return model fitness as a weighted combination of metrics
	#	mean_results	Return mean of results, mp, mr, map50, map
	#	update			Update the evaluation metrics with a new set of results

	# Predict with the model
	results = model("https://ultralytics.com/images/bus.jpg")  # A list of ultralytics.engine.results.Results

	# Export the model
	success = model.export(format="onnx")

# REF [site] >> https://docs.ultralytics.com/tasks/classification/
def yolov8_classification_example():
	from ultralytics import YOLO

	# Load a model
	if False:
		model = YOLO("yolov8n-cls.yaml")  # Build a new model from scratch
	elif True:
		model = YOLO("yolov8n-cls.pt")  # Load a pretrained model (recommended for training)
	else:
		model = YOLO("path/to/best.pt")  # Load a custom model
	#print("Model info:")
	#print(model.info())

	# Train the model
	model.train(data="mnist160", epochs=100, imgsz=64)

	# Validate the model
	metrics = model.val()  # No arguments needed, dataset and settings remembered
	#	metrics.top1  # Top1 accuracy
	#	metrics.top5  # Top5 accuracy

	# Predict with the model
	results = model("https://ultralytics.com/images/bus.jpg")  # A list of ultralytics.engine.results.Results

	# Export the model
	success = model.export(format="onnx")

# REF [site] >> https://docs.ultralytics.com/models/yolo11/
def yolo11_example():
	# Model			Filenames																			Task
	# YOLO11		yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt								Detection
	# YOLO11-seg	yolo11n-seg.pt yolo11s-seg.pt yolo11m-seg.pt yolo11l-seg.pt yolo11x-seg.pt			Instance Segmentation
	# YOLO11-pose	yolo11n-pose.pt yolo11s-pose.pt yolo11m-pose.pt yolo11l-pose.pt yolo11x-pose.pt		Pose/Keypoints
	# YOLO11-obb	yolo11n-obb.pt yolo11s-obb.pt yolo11m-obb.pt yolo11l-obb.pt yolo11x-obb.pt			Oriented Detection
	# YOLO11-cls	yolo11n-cls.pt yolo11s-cls.pt yolo11m-cls.pt yolo11l-cls.pt yolo11x-cls.pt			Classification

	if True:
		from ultralytics import YOLO

		# Load a COCO-pretrained YOLO11n model
		model = YOLO("yolo11n.pt")

		# Train the model on the COCO8 example dataset for 100 epochs
		results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

		# Run inference with the YOLO11n model on the 'bus.jpg' image
		results = model("path/to/bus.jpg")

# REF [site] >> https://docs.ultralytics.com/tasks/segment/
def yolo11_instance_segmentation_example():
	import time
	import numpy as np
	import torch
	from ultralytics import YOLO
	import cv2

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# Load a model
	# Official models
	model_name = "yolo11n-seg.pt"
	#model_name = "yolo11s-seg.pt"
	#model_name = "yolo11m-seg.pt"
	#model_name = "yolo11l-seg.pt"
	#model_name = "yolo11x-seg.pt"
	# Custom models
	#model_name = "path/to/best.pt"
	model = YOLO(model_name)

	# Predict with the model
	print("Segmenting...")
	start_time = time.time()
	results = model("https://ultralytics.com/images/bus.jpg", device=device)  # A list of ultralytics.engine.results.Results
	print(f"Segmented: {(time.time() - start_time) * 1000} msecs.")

	# ultralytics.engine.results.Results
	#	https://docs.ultralytics.com/reference/engine/results/
	#
	#	orig_img: np.ndarray			The original image as a numpy array
	#	orig_shape: tuple[int, int]		Original image shape in (height, width) format
	#	boxes: Boxes | None				Detected bounding boxes
	#	masks: Masks | None				Segmentation masks
	#	probs: Probs | None				Classification probabilities
	#	keypoints: Keypoints | None		Detected keypoints
	#	obb: OBB | None					Oriented bounding boxes
	#	speed: dict						Dictionary containing inference speed information
	#	names: dict						Dictionary mapping class indices to class names
	#	path: str						Path to the input image file
	#	save_dir: str | None			Directory to save results
	#
	# ultralytics.engine.results.Boxes
	#	data: torch.Tensor | np.ndarray		The raw tensor containing detection boxes and associated data
	#	orig_shape: tuple[int, int]			The original image dimensions (height, width)
	#	is_track: bool						Indicates whether tracking IDs are included in the box data
	#	xyxy: torch.Tensor | np.ndarray		Boxes in [x1, y1, x2, y2] format (#objects x 4)
	#	conf: torch.Tensor | np.ndarray		Confidence scores for each box
	#	cls: torch.Tensor | np.ndarray		Class labels for each box
	#	id: torch.Tensor | None				Tracking IDs for each box (if available)
	#	xywh: torch.Tensor | np.ndarray		Boxes in [x, y, width, height] format (#objects x 4)
	#	xyxyn: torch.Tensor | np.ndarray	Normalized [x1, y1, x2, y2] boxes relative to orig_shape (#objects x 4)
	#	xywhn: torch.Tensor | np.ndarray	Normalized [x, y, width, height] boxes relative to orig_shape (#objects x 4)
	# ultralytics.engine.results.Masks
	#	data: torch.Tensor | np.ndarray		The raw tensor or array containing mask data (#objects x H x W)
	#	orig_shape: tuple					Original image shape in (height, width) format
	#	xy: list[np.ndarray]				A list of segments in pixel coordinates [#objects x (#points x 2)]
	#	xyn: list[np.ndarray]				A list of normalized segments [#objects x (#points x 2)]
	# ultralytics.engine.results.Probs
	# ultralytics.engine.results.Keypoints
	# ultralytics.engine.results.OBB

	# Access the results
	for result in results:
		if result.boxes is None or result.masks is None:
			print(f"No boxes or masks detected: {result.boxes=}, {result.masks=}.")
			continue

		if True:
			for idx, (cls, mask, boundary) in enumerate(zip(result.boxes.cls, result.masks.data, result.masks.xy)):
				mask_img = result.orig_img.copy()

				mask_boundary = boundary.astype(np.int32)
				cv2.polylines(mask_img, [mask_boundary], isClosed=True, color=(0, 0, 255), thickness=2)

				cv2.imshow(f"Mask #{idx} - {result.names[int(cls)]}", mask_img)
		else:
			for idx, (cls, mask, boundary) in enumerate(zip(result.boxes.cls, result.masks.data, result.masks.xyn)):
				mask_img = cv2.cvtColor(mask.cpu().numpy() * 255, cv2.COLOR_GRAY2BGR)

				mask_boundary = boundary.copy()
				mask_boundary[:,0] *= mask.shape[1]
				mask_boundary[:,1] *= mask.shape[0]
				mask_boundary = mask_boundary.astype(np.int32)
				cv2.polylines(mask_img, [mask_boundary], isClosed=True, color=(0, 0, 255), thickness=2)

				cv2.imshow(f"Mask #{idx} - {result.names[int(cls)]}", mask_img)
			cv2.imshow("Image", result.orig_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# REF [site] >> https://docs.ultralytics.com/models/yolo12/
def yolo12_example():
	# Model Type	Task
	# YOLO12		Detection
	# YOLO12-seg	Segmentation
	# YOLO12-pose	Pose
	# YOLO12-cls	Classification
	# YOLO12-obb	OBB

	# Model
	# YOLO12n
	# YOLO12s
	# YOLO12m
	# YOLO12l
	# YOLO12x

	if True:
		from ultralytics import YOLO

		# Load a COCO-pretrained YOLO12n model
		model = YOLO("yolo12n.pt")

		# Train the model on the COCO8 example dataset for 100 epochs
		results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

		# Run inference with the YOLO12n model on the 'bus.jpg' image
		results = model("path/to/bus.jpg")

# REF [site] >> https://docs.ultralytics.com/models/yolo-nas/
def yolo_nas_example():
	# Model Type	Pre-trained Weights		Tasks Supported
	# YOLO-NAS-s	yolo_nas_s.pt			Object Detection
	# YOLO-NAS-m	yolo_nas_m.pt			Object Detection
	# YOLO-NAS-l	yolo_nas_l.pt			Object Detection

	# Model
	# YOLO-NAS S
	# YOLO-NAS M
	# YOLO-NAS L
	# YOLO-NAS S INT-8
	# YOLO-NAS M INT-8
	# YOLO-NAS L INT-8

	if True:
		# Inference and Validation Examples

		from ultralytics import NAS

		# Load a COCO-pretrained YOLO-NAS-s model
		model = NAS("yolo_nas_s.pt")

		# Display model information (optional)
		model.info()

		# Validate the model on the COCO8 example dataset
		results = model.val(data="coco8.yaml")

		# Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
		results = model("path/to/bus.jpg")

# REF [site] >> https://docs.ultralytics.com/models/yolo-world/
def yolo_world_example():
	# Model Type		Pre-trained Weights		Tasks Supported
	# YOLOv8s-world		yolov8s-world.pt		Object Detection
	# YOLOv8s-worldv2	yolov8s-worldv2.pt		Object Detection
	# YOLOv8m-world		yolov8m-world.pt		Object Detection
	# YOLOv8m-worldv2	yolov8m-worldv2.pt		Object Detection
	# YOLOv8l-world		yolov8l-world.pt		Object Detection
	# YOLOv8l-worldv2	yolov8l-worldv2.pt		Object Detection
	# YOLOv8x-world		yolov8x-world.pt		Object Detection
	# YOLOv8x-worldv2	yolov8x-worldv2.pt		Object Detection

	if True:
		# Train Usage

		from ultralytics import YOLOWorld

		# Load a pretrained YOLOv8s-worldv2 model
		model = YOLOWorld("yolov8s-worldv2.pt")

		# Train the model on the COCO8 example dataset for 100 epochs
		results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

		# Run inference with the YOLOv8n model on the 'bus.jpg' image
		results = model("path/to/bus.jpg")

	if True:
		# Predict Usage

		from ultralytics import YOLOWorld

		# Initialize a YOLO-World model
		model = YOLOWorld("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

		# Execute inference with the YOLOv8s-world model on the specified image
		results = model.predict("path/to/image.jpg")

		# Show results
		results[0].show()

	if True:
		# Val Usage

		from ultralytics import YOLO

		# Create a YOLO-World model
		model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

		# Conduct model validation on the COCO8 example dataset
		metrics = model.val(data="coco8.yaml")

	if True:
		# Track Usage

		from ultralytics import YOLO

		# Create a YOLO-World model
		model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

		# Track with a YOLO-World model on a video
		results = model.track(source="path/to/video.mp4")

	if True:
		# Custom Inference Prompts

		from ultralytics import YOLO

		# Initialize a YOLO-World model
		model = YOLO("yolov8s-world.pt")  # or choose yolov8m/l-world.pt

		# Define custom classes
		model.set_classes(["person", "bus"])

		# Execute prediction for specified categories on an image
		results = model.predict("path/to/image.jpg")

		# Show results
		results[0].show()

	if True:
		# Persisting Models with Custom Vocabulary

		# First load a YOLO-World model, set custom classes for it and save it:
		from ultralytics import YOLO

		# Initialize a YOLO-World model
		model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt

		# Define custom classes
		model.set_classes(["person", "bus"])

		# Save the model with the defined offline vocabulary
		model.save("./custom_yolov8s.pt")

		# After saving, the custom_yolov8s.pt model behaves like any other pre-trained YOLOv8 model but with a key difference: it is now optimized to detect only the classes you have defined.
		# This customization can significantly improve detection performance and efficiency for your specific application scenarios.

		# Load your custom model
		model = YOLO("./custom_yolov8s.pt")

		# Run inference to detect your custom classes
		results = model.predict("path/to/image.jpg")

		# Show results
		results[0].show()

	if True:
		# Launch training from scratch

		from ultralytics import YOLOWorld
		from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

		data = dict(
			train=dict(
				yolo_data=["./Objects365.yaml"],
				grounding_data=[
					dict(
						img_path="flickr30k/images",
						json_file="flickr30k/final_flickr_separateGT_train.json",
					),
					dict(
						img_path="GQA/images",
						json_file="GQA/final_mixed_train_no_coco.json",
					),
				],
			),
			val=dict(yolo_data=["lvis.yaml"]),
		)
		model = YOLOWorld("yolov8s-worldv2.yaml")
		model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)

# REF [site] >> https://docs.ultralytics.com/models/yolo-world/
def yolo_world_test():
	import time
	from ultralytics import YOLOWorld
	import cv2

	input_file = "./bus.jpg"

	# Initialize a YOLO-World model
	#model_name = "yolov8s-world.pt"
	#model_name = "yolov8m-world.pt"
	#model_name = "yolov8l-world.pt"
	#model_name = "yolov8x-world.pt"
	model_name = "yolov8s-worldv2.pt"
	#model_name = "yolov8m-worldv2.pt"
	#model_name = "yolov8l-worldv2.pt"
	#model_name = "yolov8x-worldv2.pt"
	model = YOLOWorld(model_name)

	custom_classes = [
		"person",
		"bus",
	]
	model.set_classes(custom_classes)

	# Detect
	print("Detecting...")
	start_time = time.time()
	#results = model.predict(input_file)
	results = model.predict(input_file, conf=0.5, verbose=False)
	print(f"Detected (#objects = {len(results[0])}): {(time.time() - start_time) * 1000} msecs.")

	# Show results
	print(f"Box classes: {results[0].boxes.cls}.")
	print(f"Box confidences: {results[0].boxes.conf}")

	#results[0].show()
	annotated_frame = results[0].plot()  # numpy.array, (height, width, channels), BGR
	cv2.imshow("YOLO-World", annotated_frame)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Export the model
	if False:
		success = model.export(format="onnx", opset=12, imgsz=640)

		print(f"Export success: {success}.")

# REF [site] >> https://docs.ultralytics.com/models/yoloe/
def yoloe_example():
	# Text/Visual Prompt models
	#	Model Type		Pre-trained Weights		Tasks Supported
	#	YOLOE-11S		yoloe-11s-seg.pt		Instance Segmentation
	#	YOLOE-11M		yoloe-11m-seg.pt		Instance Segmentation
	#	YOLOE-11L		yoloe-11l-seg.pt		Instance Segmentation
	#	YOLOE-v8S		yoloe-v8s-seg.pt		Instance Segmentation
	#	YOLOE-v8M		yoloe-v8m-seg.pt		Instance Segmentation
	#	YOLOE-v8L		yoloe-v8l-seg.pt		Instance Segmentation

	# Prompt Free models
	#	Model Type		Pre-trained Weights		Tasks Supported
	#	YOLOE-11S-PF	yoloe-11s-seg-pf.pt		Instance Segmentation
	#	YOLOE-11M-PF	yoloe-11m-seg-pf.pt		Instance Segmentation
	#	YOLOE-11L-PF	yoloe-11l-seg-pf.pt		Instance Segmentation
	#	YOLOE-v8S-PF	yoloe-v8s-seg-pf.pt		Instance Segmentation
	#	YOLOE-v8M-PF	yoloe-v8m-seg-pf.pt		Instance Segmentation
	#	YOLOE-v8L-PF	yoloe-v8l-seg-pf.pt		Instance Segmentation

	#-----
	# Train Usage
	#	Fine-Tuning on custom dataset

	if True:
		# Instance segmentation (Fine-Tuning)

		from ultralytics import YOLOE
		from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer as Trainer

		model = YOLOE("yoloe-11s-seg.pt")

		results = model.train(
			data="coco128-seg.yaml",
			epochs=80,
			close_mosaic=10,
			batch=16,
			optimizer="AdamW",
			lr0=1e-3,
			warmup_bias_lr=0.0,
			weight_decay=0.025,
			momentum=0.9,
			workers=4,
			device="0",
			trainer=Trainer,
		)

	if True:
		# Instance segmentation (Fine-Tuning)

		from ultralytics import YOLOE
		from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer as Trainer

		model = YOLOE("yoloe-11s-seg.pt")
		head_index = len(model.model.model) - 1
		freeze = [str(f) for f in range(0, head_index)]  # freeze all layers except head

		# Freeze all head components except classification branch
		for name, child in model.model.model[-1].named_children():
			if "cv3" not in name:
				freeze.append(f"{head_index}.{name}")

		# Freeze detection branch components
		freeze.extend([
			f"{head_index}.cv3.0.0",
			f"{head_index}.cv3.0.1",
			f"{head_index}.cv3.1.0",
			f"{head_index}.cv3.1.1",
			f"{head_index}.cv3.2.0",
			f"{head_index}.cv3.2.1",
		])

		results = model.train(
			data="coco128-seg.yaml",
			epochs=2,
			close_mosaic=0,
			batch=16,
			optimizer="AdamW",
			lr0=1e-3,
			warmup_bias_lr=0.0,
			weight_decay=0.025,
			momentum=0.9,
			workers=4,
			device="0",
			trainer=Trainer,
			freeze=freeze,
		)

	if True:
		# Object detection

		from ultralytics import YOLOE
		from ultralytics.models.yolo.yoloe import YOLOEPETrainer as Trainer  # noqa

		# Create detection model from yaml then load segmentation weights
		model = YOLOE("yoloe-11s.yaml").load("yoloe-11s-seg.pt")
		# Rest of the code is same as the instance segmentation fine-tuning example above

	#-----
	# Predict Usage

	if True:
		# Text Prompt

		from ultralytics import YOLOE

		# Initialize a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

		# Set text prompt to detect person and bus. You only need to do this once after you load the model.
		names = ["person", "bus"]
		model.set_classes(names, model.get_text_pe(names))

		# Run detection on the given image
		results = model.predict("path/to/image.jpg")

		# Show results
		results[0].show()

	if True:
		# Visual Prompt

		import numpy as np
		from ultralytics import YOLOE
		from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

		# Initialize a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")

		# Define visual prompts using bounding boxes and their corresponding class IDs.
		# Each box highlights an example of the object you want the model to detect.
		visual_prompts = dict(
			bboxes=np.array(
				[
					[221.52, 405.8, 344.98, 857.54],  # Box enclosing person
					[120, 425, 160, 445],  # Box enclosing glasses
				],
			),
			cls=np.array(
				[
					0,  # ID to be assigned for person
					1,  # ID to be assigned for glassses
				]
			),
		)

		# Run inference on an image, using the provided visual prompts as guidance
		results = model.predict(
			"ultralytics/assets/bus.jpg",
			visual_prompts=visual_prompts,
			predictor=YOLOEVPSegPredictor,
		)

		# Show results
		results[0].show()

	if True:
		# Prompt free

		from ultralytics import YOLOE

		# Initialize a YOLOE model
		model = YOLOE("yoloe-11l-seg-pf.pt")

		# Run prediction. No prompts required.
		results = model.predict("path/to/image.jpg")

		# Show results
		results[0].show()

	#-----
	# Val Usage

	if True:
		# Text Prompt

		from ultralytics import YOLOE

		# Create a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

		# Conduct model validation on the COCO128-seg example dataset
		metrics = model.val(data="coco128-seg.yaml")

	if True:
		# Visual Prompt
		#	It's using the provided dataset to extract visual embeddings for each category

		from ultralytics import YOLOE

		# Create a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

		# Conduct model validation on the COCO128-seg example dataset
		metrics = model.val(data="coco128-seg.yaml", load_vp=True)

	if True:
		# Visual Prompt
		#	Alternatively we could use another dataset as a reference dataset to extract visual embeddings for each category.
		#	Note this reference dataset should have exactly the same categories as provided dataset.

		from ultralytics import YOLOE

		# Create a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

		# Conduct model validation on the COCO128-seg example dataset
		metrics = model.val(data="coco128-seg.yaml", load_vp=True, refer_data="coco.yaml")

	if True:
		# Prompt free

		from ultralytics import YOLOE

		# Create a YOLOE model
		model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

		# Conduct model validation on the COCO128-seg example dataset
		metrics = model.val(data="coco128-seg.yaml")

	#-----
	# Launching training from scratch
	#	Visual Prompt models are fine-tuned based on trained-well Text Prompt models

	if True:
		# Text Prompt

		from ultralytics import YOLOE
		from ultralytics.models.yolo.yoloe import YOLOESegTrainerFromScratch

		data = dict(
			train=dict(
				yolo_data=["./Objects365.yaml"],
				grounding_data=[
					dict(
						img_path="flickr/full_images/",
						json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
					),
					dict(
						img_path="mixed_grounding/gqa/images",
						json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
					),
				],
			),
			val=dict(yolo_data=["lvis.yaml"]),
		)

		model = YOLOE("yoloe-11l-seg.yaml")
		model.train(
			data=data,
			batch=128,
			epochs=30,
			close_mosaic=2,
			optimizer="AdamW",
			lr0=2e-3,
			warmup_bias_lr=0.0,
			weight_decay=0.025,
			momentum=0.9,
			workers=4,
			trainer=YOLOESegTrainerFromScratch,
			device="0,1,2,3,4,5,6,7",
		)

	if True:
		# Visual Prompt

		from ultralytics import YOLOE
		from ultralytics.utils.patches import torch_load

		det_model = YOLOE("yoloe-11l.yaml")
		state = torch_load("yoloe-11l-seg.pt")
		det_model.load(state["model"])
		det_model.save("yoloe-11l-seg-det.pt")

		# Start training
		from ultralytics.models.yolo.yoloe import YOLOESegVPTrainer

		data = dict(
			train=dict(
				yolo_data=["./Objects365.yaml"],
				grounding_data=[
					dict(
						img_path="flickr/full_images/",
						json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
					),
					dict(
						img_path="mixed_grounding/gqa/images",
						json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
					),
				],
			),
			val=dict(yolo_data=["lvis.yaml"]),
		)

		model = YOLOE("yoloe-11l-seg.pt")
		# Replace to yoloe-11l-seg-det.pt if converted to detection model
		#model = YOLOE("yoloe-11l-seg-det.pt")

		# Freeze every layer except of the savpe module.
		head_index = len(model.model.model) - 1
		freeze = list(range(0, head_index))
		for name, child in model.model.model[-1].named_children():
			if "savpe" not in name:
				freeze.append(f"{head_index}.{name}")

		model.train(
			data=data,
			batch=128,
			epochs=2,
			close_mosaic=2,
			optimizer="AdamW",
			lr0=16e-3,
			warmup_bias_lr=0.0,
			weight_decay=0.025,
			momentum=0.9,
			workers=4,
			trainer=YOLOESegVPTrainer,  # Use YOLOEVPTrainer if converted to detection model
			device="0,1,2,3,4,5,6,7",
			freeze=freeze,
		)

		# Convert back to segmentation model after training. Only needed if you converted segmentation model to detection model before training
		from copy import deepcopy

		model = YOLOE("yoloe-11l-seg.yaml")
		model.load("yoloe-11l-seg.pt")

		vp_model = YOLOE("yoloe-11l-vp.pt")
		model.model.model[-1].savpe = deepcopy(vp_model.model.model[-1].savpe)
		model.eval()
		model.save("yoloe-11l-seg.pt")

	if True:
		# Prompt free

		from ultralytics import YOLOE
		from ultralytics.utils.patches import torch_load

		det_model = YOLOE("yoloe-11l.yaml")
		state = torch_load("yoloe-11l-seg.pt")
		det_model.load(state["model"])
		det_model.save("yoloe-11l-seg-det.pt")

		# Start training
		data = dict(
			train=dict(
				yolo_data=["./Objects365.yaml"],
				grounding_data=[
					dict(
						img_path="flickr/full_images/",
						json_file="flickr/annotations/final_flickr_separateGT_train_segm.json",
					),
					dict(
						img_path="mixed_grounding/gqa/images",
						json_file="mixed_grounding/annotations/final_mixed_train_no_coco_segm.json",
					),
				],
			),
			val=dict(yolo_data=["lvis.yaml"]),
		)

		model = YOLOE("yoloe-11l-seg.pt")
		# Replace to yoloe-11l-seg-det.pt if converted to detection model
		#model = YOLOE("yoloe-11l-seg-det.pt")

		# Freeze layers
		head_index = len(model.model.model) - 1
		freeze = [str(f) for f in range(0, head_index)]
		for name, child in model.model.model[-1].named_children():
			if "cv3" not in name:
				freeze.append(f"{head_index}.{name}")

		freeze.extend([
			f"{head_index}.cv3.0.0",
			f"{head_index}.cv3.0.1",
			f"{head_index}.cv3.1.0",
			f"{head_index}.cv3.1.1",
			f"{head_index}.cv3.2.0",
			f"{head_index}.cv3.2.1",
		])

		model.train(
			data=data,
			batch=128,
			epochs=1,
			close_mosaic=1,
			optimizer="AdamW",
			lr0=2e-3,
			warmup_bias_lr=0.0,
			weight_decay=0.025,
			momentum=0.9,
			workers=4,
			trainer=YOLOEPEFreeTrainer,
			device="0,1,2,3,4,5,6,7",
			freeze=freeze,
			single_cls=True,  # This is needed
		)

		# Convert back to segmentation model after training. Only needed if you converted segmentation model to detection model before training.
		from copy import deepcopy

		model = YOLOE("yoloe-11l-seg.pt")
		model.eval()

		pf_model = YOLOE("yoloe-11l-seg-pf.pt")
		names = ["object"]
		tpe = model.get_text_pe(names)
		model.set_classes(names, tpe)
		model.model.model[-1].fuse(model.model.pe)

		model.model.model[-1].cv3[0][2] = deepcopy(pf_model.model.model[-1].cv3[0][2]).requires_grad_(True)
		model.model.model[-1].cv3[1][2] = deepcopy(pf_model.model.model[-1].cv3[1][2]).requires_grad_(True)
		model.model.model[-1].cv3[2][2] = deepcopy(pf_model.model.model[-1].cv3[2][2]).requires_grad_(True)
		del model.model.pe
		model.save("yoloe-11l-seg-pf.pt")

	if True:
		# Training and Inference

		from ultralytics import YOLO

		# Load pre-trained YOLOE model and train on custom data
		model = YOLO("yoloe-11s-seg.pt")
		model.train(data="path/to/data.yaml", epochs=50, imgsz=640)

		# Run inference using text prompts ("person", "bus")
		model.set_classes(["person", "bus"])
		results = model.predict(source="test_images/street.jpg")
		results[0].save()  # Save annotated output

# REF [site] >> https://docs.ultralytics.com/models/rtdetr/
def rt_detr_example():
	# Model Type			Pre-trained Weights		Tasks Supported
	# RT-DETR Large			rtdetr-l.pt				Object Detection
	# RT-DETR Extra-Large	rtdetr-x.pt				Object Detection

	if True:
		from ultralytics import RTDETR

		# Load a COCO-pretrained RT-DETR-l model
		model = RTDETR("rtdetr-l.pt")

		# Display model information (optional)
		model.info()

		# Train the model on the COCO8 example dataset for 100 epochs
		results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

		# Run inference with the RT-DETR-l model on the 'bus.jpg' image
		results = model("path/to/bus.jpg")

# REF [site] >> https://docs.ultralytics.com/models/sam/
def sam_example():
	# Model type	Pre-trained Weights		Tasks Supported
	# SAM base		sam_b.pt				Instance Segmentation
	# SAM large		sam_l.pt				Instance Segmentation

	if True:
		# Segment with prompts
		#	Segment image with given prompts

		from ultralytics import SAM

		# Load a model
		model = SAM("sam_b.pt")

		# Display model information (optional)
		model.info()

		# Run inference with bboxes prompt
		results = model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

		# Run inference with single point
		results = model(points=[900, 370], labels=[1])

		# Run inference with multiple points
		results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

		# Run inference with multiple points prompt per object
		results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

		# Run inference with negative points prompt
		results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

	if True:
		# Segment everything
		#	Segment the whole image

		from ultralytics import SAM

		# Load a model
		model = SAM("sam_b.pt")

		# Display model information (optional)
		model.info()

		# Run inference
		model("path/to/image.jpg")

	#-----
	# SAMPredictor example
	#	This way you can set image once and run prompts inference multiple times without running image encoder multiple times

	if True:
		# Prompt inference

		from ultralytics.models.sam import Predictor as SAMPredictor
		import cv2

		# Create SAMPredictor
		overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
		predictor = SAMPredictor(overrides=overrides)

		# Set image
		predictor.set_image("ultralytics/assets/zidane.jpg")  # set with image file
		predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # set with np.ndarray
		results = predictor(bboxes=[439, 437, 524, 709])

		# Run inference with single point prompt
		results = predictor(points=[900, 370], labels=[1])

		# Run inference with multiple points prompt
		results = predictor(points=[[400, 370], [900, 370]], labels=[[1, 1]])

		# Run inference with negative points prompt
		results = predictor(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

		# Reset image
		predictor.reset_image()

	if True:
		# Segment everything

		from ultralytics.models.sam import Predictor as SAMPredictor

		# Create SAMPredictor
		overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
		predictor = SAMPredictor(overrides=overrides)

		# Segment with additional args
		results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)

	if True:
		# SAM Comparison vs YOLO

		from ultralytics import ASSETS, SAM, YOLO, FastSAM

		# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
		for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
			model = SAM(file)
			model.info()
			model(ASSETS)

		# Profile FastSAM-s
		model = FastSAM("FastSAM-s.pt")
		model.info()
		model(ASSETS)

		# Profile YOLO models
		for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
			model = YOLO(file_name)
			model.info()
			model(ASSETS)

	if True:
		# Auto-Annotation: A Quick Path to Segmentation Datasets
		#	Generate Your Segmentation Dataset Using a Detection Model

		from ultralytics.data.annotator import auto_annotate

		auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam_b.pt")

# REF [site] >> https://docs.ultralytics.com/models/sam-2/
def sam2_example():
	# Model Type		Pre-trained Weights		Tasks Supported
	# SAM 2 tiny		sam2_t.pt				Instance Segmentation
	# SAM 2 small		sam2_s.pt				Instance Segmentation
	# SAM 2 base		sam2_b.pt				Instance Segmentation
	# SAM 2 large		sam2_l.pt				Instance Segmentation
	# SAM 2.1 tiny		sam2.1_t.pt				Instance Segmentation
	# SAM 2.1 small		sam2.1_s.pt				Instance Segmentation
	# SAM 2.1 base		sam2.1_b.pt				Instance Segmentation
	# SAM 2.1 large		sam2.1_l.pt				Instance Segmentation

	if True:
		# Segment with Prompts
		#	Use prompts to segment specific objects in images or videos

		from ultralytics import SAM

		# Load a model
		model = SAM("sam2.1_b.pt")

		# Display model information (optional)
		model.info()

		# Run inference with bboxes prompt
		results = model("path/to/image.jpg", bboxes=[100, 100, 200, 200])

		# Run inference with single point
		results = model(points=[900, 370], labels=[1])

		# Run inference with multiple points
		results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

		# Run inference with multiple points prompt per object
		results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

		# Run inference with negative points prompt
		results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

	if True:
		# Segment Everything
		#	Segment the entire image or video content without specific prompts

		from ultralytics import SAM

		# Load a model
		model = SAM("sam2.1_b.pt")

		# Display model information (optional)
		model.info()

		# Run inference
		model("path/to/video.mp4")

	if True:
		# Segment Video and Track objects
		#	Segment the entire video content with specific prompts and track objects

		from ultralytics.models.sam import SAM2VideoPredictor

		# Create SAM2VideoPredictor
		overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam2_b.pt")
		predictor = SAM2VideoPredictor(overrides=overrides)

		# Run inference with single point
		results = predictor(source="path/to/test.mp4", points=[920, 470], labels=[1])

		# Run inference with multiple points
		results = predictor(source="path/to/test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

		# Run inference with multiple points prompt per object
		results = predictor(source="path/to/test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

		# Run inference with negative points prompt
		results = predictor(source="path/to/test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])

	if True:
		# SAM 2 Comparison vs YOLO

		from ultralytics import ASSETS, SAM, YOLO, FastSAM

		# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
		for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
			model = SAM(file)
			model.info()
			model(ASSETS)

		# Profile FastSAM-s
		model = FastSAM("FastSAM-s.pt")
		model.info()
		model(ASSETS)

		# Profile YOLO models
		for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
			model = YOLO(file_name)
			model.info()
			model(ASSETS)

	if True:
		# Auto-Annotation: Efficient Dataset Creation

		from ultralytics.data.annotator import auto_annotate

		auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam2_b.pt")

# REF [site] >> https://docs.ultralytics.com/models/mobile-sam/
def mobile_sam_example():
	# Model Type	Pre-trained Weights		Tasks Supported
	# MobileSAM		mobile_sam.pt			Instance Segmentation

	if True:
		# MobileSAM Comparison vs YOLO

		from ultralytics import ASSETS, SAM, YOLO, FastSAM

		# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
		for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
			model = SAM(file)
			model.info()
			model(ASSETS)

		# Profile FastSAM-s
		model = FastSAM("FastSAM-s.pt")
		model.info()
		model(ASSETS)

		# Profile YOLO models
		for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
			model = YOLO(file_name)
			model.info()
			model(ASSETS)

	if True:
		# Point Prompt

		from ultralytics import SAM

		# Load the model
		model = SAM("mobile_sam.pt")

		# Predict a segment based on a single point prompt
		model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

		# Predict multiple segments based on multiple points prompt
		model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

		# Predict a segment based on multiple points prompt per object
		model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

		# Predict a segment using both positive and negative prompts.
		model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

	if True:
		# Box Prompt (???)

		from ultralytics import SAM

		# Load the model
		model = SAM("mobile_sam.pt")

		# Predict a segment based on a single point prompt
		model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

		# Predict multiple segments based on multiple points prompt
		model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

		# Predict a segment based on multiple points prompt per object
		model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

		# Predict a segment using both positive and negative prompts.
		model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])

	if True:
		# Automatically Build Segmentation Datasets Using a Detection Model

		from ultralytics.data.annotator import auto_annotate

		auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="mobile_sam.pt")

# REF [site] >> https://docs.ultralytics.com/models/fast-sam/
def fast_sam_example():
	# Model Type	Pre-trained Weights		Tasks Supported
	# FastSAM-s		FastSAM-s.pt			Instance Segmentation
	# FastSAM-x		FastSAM-x.pt			Instance Segmentation

	if True:
		# FastSAM Comparison vs YOLO

		from ultralytics import ASSETS, SAM, YOLO, FastSAM

		# Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
		for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
			model = SAM(file)
			model.info()
			model(ASSETS)

		# Profile FastSAM-s
		model = FastSAM("FastSAM-s.pt")
		model.info()
		model(ASSETS)

		# Profile YOLO models
		for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
			model = YOLO(file_name)
			model.info()
			model(ASSETS)

	#-----
	# Predict Usage

	if True:
		# Perform object detection on an image

		from ultralytics import FastSAM

		# Define an inference source
		source = "path/to/bus.jpg"

		# Create a FastSAM model
		model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

		# Run inference on an image
		everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

		# Run inference with bboxes prompt
		results = model(source, bboxes=[439, 437, 524, 709])

		# Run inference with points prompt
		results = model(source, points=[[200, 200]], labels=[1])

		# Run inference with texts prompt
		results = model(source, texts="a photo of a dog")

		# Run inference with bboxes and points and texts prompt at the same time
		results = model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")

	if True:
		# FastSAMPredictor example
		#	 This way you can run inference on image and get all the segment results once and run prompts inference multiple times without running inference multiple times

		from ultralytics.models.fastsam import FastSAMPredictor

		# Create FastSAMPredictor
		overrides = dict(conf=0.25, task="segment", mode="predict", model="FastSAM-s.pt", save=False, imgsz=1024)
		predictor = FastSAMPredictor(overrides=overrides)

		# Segment everything
		everything_results = predictor("ultralytics/assets/bus.jpg")

		# Prompt inference
		bbox_results = predictor.prompt(everything_results, bboxes=[[200, 200, 300, 300]])
		point_results = predictor.prompt(everything_results, points=[200, 200])
		text_results = predictor.prompt(everything_results, texts="a photo of a dog")

	#-----
	# Val Usage

	if True:
		# Validation of the model on a dataset

		from ultralytics import FastSAM

		# Create a FastSAM model
		model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

		# Validate the model
		results = model.val(data="coco8-seg.yaml")

	#-----
	# Track Usage

	if True:
		# Perform object tracking on an image

		from ultralytics import FastSAM

		# Create a FastSAM model
		model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

		# Track with a FastSAM model on a video
		results = model.track(source="path/to/video.mp4", imgsz=640)

def visualize_ultralytics_results(results):
	import numpy as np
	import cv2

	for result in results:
		if result.boxes is None or result.masks is None:
			print(f"No boxes or masks detected: {result.boxes=}, {result.masks=}.")
			continue

		#result.summary()
		#result.show()
		#annotated_frame = result.plot()  # numpy.array, (height, width, channels), BGR
		#cv2.imshow("YOLO-World", annotated_frame)

		if True:
			for idx, (cls, mask, boundary) in enumerate(zip(result.boxes.cls, result.masks.data, result.masks.xy)):
				mask_img = result.orig_img.copy()

				mask_boundary = boundary.astype(np.int32)
				cv2.polylines(mask_img, [mask_boundary], isClosed=True, color=(0, 0, 255), thickness=2)

				cv2.imshow(f"Mask #{idx} - {result.names[int(cls)]}", mask_img)
		else:
			for idx, (cls, mask, boundary) in enumerate(zip(result.boxes.cls, result.masks.data, result.masks.xyn)):
				mask_img = cv2.cvtColor(mask.cpu().numpy() * 255, cv2.COLOR_GRAY2BGR)

				mask_boundary = boundary.copy()
				mask_boundary[:,0] *= mask.shape[1]
				mask_boundary[:,1] *= mask.shape[0]
				mask_boundary = mask_boundary.astype(np.int32)
				cv2.polylines(mask_img, [mask_boundary], isClosed=True, color=(0, 0, 255), thickness=2)

				cv2.imshow(f"Mask #{idx} - {result.names[int(cls)]}", mask_img)
			cv2.imshow(f"Image", result.orig_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# REF [site] >> https://docs.ultralytics.com/models/fast-sam/#predict-usage
def fast_sam_test():
	import time
	from ultralytics import FastSAM

	input_file = "./bus.jpg"

	# Create a FastSAM model
	#model = FastSAM("FastSAM-s.pt")
	model = FastSAM("FastSAM-x.pt")

	# Run inference on an image
	print("Segmenting...")
	start_time = time.time()
	everything_results = model(input_file, device="cpu", retina_masks=True, imgsz=448, conf=0.4, iou=0.9)
	print(f"Segmented (#objects = {len(everything_results[0])}): {(time.time() - start_time) * 1000} msecs.")

	# Visualize the results
	visualize_ultralytics_results(everything_results)

	# Export the model
	if False:
		success = model.export(format="onnx", opset=12, imgsz=448)

		print(f"Export success: {success}.")

# REF [site] >> https://docs.ultralytics.com/usage/simple-utilities/
def auto_annotate_example():
	from ultralytics.data.annotator import auto_annotate

	auto_annotate(
		data="path/to/new/data",
		det_model="yolo11n.pt",
		sam_model="mobile_sam.pt",
		device="cuda",
		output_dir="path/to/save_labels",
	)

# REF [site] >> https://docs.ultralytics.com/usage/simple-utilities/
def visualize_image_annotations_example():
	from ultralytics.data.utils import visualize_image_annotations

	label_map = {  # Define the label map with all annotated class labels
		0: "person",
		1: "car",
	}

	# Visualize
	visualize_image_annotations(
		"path/to/image.jpg",  # Input image path
		"path/to/annotations.txt",  # Annotation file path for the image
		label_map,
	)

# REF [site] >> https://docs.ultralytics.com/usage/simple-utilities/
def convert_coco_into_yolo_format_example():
	from ultralytics.data.converter import convert_coco

	convert_coco(
		"coco/annotations/",
		use_segments=False,
		use_keypoints=False,
		cls91to80=True,
	)

def main():
	# Models: https://docs.ultralytics.com/models/
	#	https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models
	#
	#	YOLOv3 ~ YOLOv12
	#		YOLOv5:
	#			Detection: yolov5nu, yolov5su, yolov5mu, yolov5lu, yolov5xu
	#		YOLOv8:
	#			Detection: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
	#			Instance segmentation: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
	#			Classification: yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls
	#			Pose estimation: yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose
	#			Oriented bounding boxes (OBB): yolov8n-obb, yolov8s-obb, yolov8m-obb, yolov8l-obb, yolov8x-obb
	#		YOLOv11:
	#			Detection: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
	#			Instance segmentation: yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg
	#			Classification: yolo11n-cls, yolo11s-cls, yolo11m-cls, yolo11l-cls, yolo11x-cls
	#			Pose estimation: yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose
	#			Oriented bounding boxes (OBB): yolo11n-obb, yolo11s-obb, yolo11m-obb, yolo11l-obb, yolo11x-obb
	#	YOLO26
	#	YOLO-NAS, YOLO-World, YOLOE
	#	Segment Anything Model (SAM), Segment Anything Model 2 (SAM 2), Segment Anything Model 3 (SAM 3), Mobile Segment Anything Model (MobileSAM), Fast Segment Anything Model (FastSAM)
	#	Realtime Detection Transformers (RT-DETR)

	# Tasks: https://docs.ultralytics.com/tasks/
	#	Object Detection: https://docs.ultralytics.com/tasks/detect/
	#	Instance Segmentation: https://docs.ultralytics.com/tasks/segment/
	#	Image Classification: https://docs.ultralytics.com/tasks/classify/
	#	Pose Estimation: https://docs.ultralytics.com/tasks/pose/
	#	Oriented Bounding Boxes Object Detection: https://docs.ultralytics.com/tasks/obb/

	# Modes: https://docs.ultralytics.com/modes/
	#	Model Training: https://docs.ultralytics.com/modes/train/
	#	Model Validation: https://docs.ultralytics.com/modes/val/
	#	Model Prediction: https://docs.ultralytics.com/modes/predict/
	#	Model Export: https://docs.ultralytics.com/modes/export/
	#	Multi-Object Tracking: https://docs.ultralytics.com/modes/track/
	#	Model Benchmarking: https://docs.ultralytics.com/modes/benchmark/

	# Datasets: https://docs.ultralytics.com/datasets/
	# Solutions: https://docs.ultralytics.com/solutions/
	# Tutorials: https://docs.ultralytics.com/guides/
	# Integrations: https://docs.ultralytics.com/integrations/

	# HUB: https://docs.ultralytics.com/hub/
	#	https://www.ultralytics.com/hub

	#-----
	#benchmark()

	#export_models_test()

	#-----
	# You Only Look Once (YOLO)
	#	Detection
	#	Instance segmentation
	#	Classification

	# YOLOv5
	#yolov5_detection_example()

	# YOLOv8
	#yolov8_detection_example()
	#yolov8_instance_segmentation_example()
	#yolov8_classification_example()

	# YOLOv11
	#yolo11_example()
	yolo11_instance_segmentation_example()  # Visualize the results

	# YOLOv12
	#yolo12_example()

	# YOLO-NAS
	#yolo_nas_example()

	# YOLO-World: Real-Time Open-Vocabulary Object Detection
	#yolo_world_example()
	#yolo_world_test()

	# YOLOE: Real-Time Seeing Anything
	#yoloe_example()

	#-----
	# Detection Transformer (DETR)

	#rt_detr_example()  # RT-DETR: Real-Time Detection Transformer

	#-----
	# Segment Anything (SAM)

	# NOTE [caution] >> Low level of completion
	#	Exporting a SAM to ONNX are not supported

	#sam_example()  # SAM, MobileSAM
	#sam2_example()  # SAM 2
	#mobile_sam_example()  # MobileSAM
	#fast_sam_example()  # FastSAM

	#-----
	# ONNX
	#	https://docs.ultralytics.com/integrations/onnx/

	# Annotation
	#	Auto-Annotation: A Quick Path to Segmentation Datasets:
	#		https://docs.ultralytics.com/models/sam/#auto-annotation-a-quick-path-to-segmentation-datasets
	#	Auto-Annotation: Efficient Dataset Creation:
	#		https://docs.ultralytics.com/models/sam-2/#auto-annotation-efficient-dataset-creation
	#	Auto Annotation with Meta's Segment Anything 2 Model using Ultralytics:
	#		https://youtu.be/M7xWw4Iodhg

	#-----
	# TensorRT
	#	https://docs.ultralytics.com/integrations/tensorrt/

	#-----
	# Simple utilities: https://docs.ultralytics.com/usage/simple-utilities/
	#	Data:
	#		Auto Labeling / Annotations
	#		Visualize Dataset Annotations
	#		Convert Segmentation Masks into YOLO Format
	#		Convert COCO into YOLO Format
	#		Get Bounding Box Dimensions
	#		Convert Bounding Boxes to Segments
	#		Convert Segments to Bounding Boxes
	#	Utilities:
	#		Image Compression
	#		Auto-split Dataset
	#		Segment-polygon to Binary Mask
	#	Bounding Boxes:
	#		Bounding Box (Horizontal) Instances
	#		Scaling Boxes
	#		Bounding Box Format Conversions
	#		All Bounding Box Conversions
	#	Plotting:
	#		Box Annotation
	#		Sweep Annotation
	#		Adaptive label Annotation
	#	Miscellaneous:
	#		Code Profiling
	#		Ultralytics Supported Formats
	#		Make Divisible

	#auto_annotate_example()
	#visualize_image_annotations_example()
	#convert_coco_into_yolo_format_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
