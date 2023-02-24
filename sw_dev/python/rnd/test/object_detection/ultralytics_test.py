#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from ultralytics import YOLO

# REF [site] >>
#	https://github.com/ultralytics/ultralytics
#	https://docs.ultralytics.com/tasks/detection/
def yolov8_detection_example():
	# Load a model.
	if False:
		model = YOLO("yolov8n.yaml")  # Build a new model from scratch.
	elif True:
		model = YOLO("yolov8n.pt")  # Load a pretrained model (recommended for training).
	else:
		model = YOLO("/path/to/best.pt")  # Load a custom model.
	#print("Model info:")
	#print(model.info())

	# Train the model.
	#model.train(data="coco128.yaml", epochs=3)
	model.train(data="coco128.yaml", epochs=100, imgsz=640)

	# Validate the model.
	metrics = model.val()  # No arguments needed, dataset and settings remembered.
	#	metrics.box.map  # mAP50-95.
	#	metrics.box.map50  # mAP50.
	#	metrics.box.map75  # mAP75.
	#	metrics.box.maps  # A list contains mAP50-95 of each category.

	# Predict with the model.
	results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image.
	
	# Export the model.
	success = model.export(format="onnx")

def yolov5_detection_example():
	# Load a model.
	if False:
		model = YOLO("yolov5nu.yaml")  # Build a new model from scratch.
	elif True:
		model = YOLO("yolov5nu.pt")  # Load a pretrained model (recommended for training).
	else:
		model = YOLO("/path/to/best.pt")  # Load a custom model.
	#print("Model info:")
	#print(model.info())

	# Train the model.
	#model.train(data="coco128.yaml", epochs=3)
	model.train(data="coco128.yaml", epochs=100, imgsz=640)

	# Validate the model.
	metrics = model.val()  # No arguments needed, dataset and settings remembered.
	#	metrics.box.map  # mAP50-95.
	#	metrics.box.map50  # mAP50.
	#	metrics.box.map75  # mAP75.
	#	metrics.box.maps  # A list contains mAP50-95 of each category.

	# Predict with the model.
	results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image.
	
	# Export the model.
	success = model.export(format="onnx")

# REF [site] >> https://docs.ultralytics.com/tasks/segmentation/
def yolov8_segmentation_example():
	# Load a model.
	if False:
		model = YOLO("yolov8n-seg.yaml")  # Build a new model from scratch.
	elif True:
		model = YOLO("yolov8n-seg.pt")  # Load a pretrained model (recommended for training).
	else:
		model = YOLO("/path/to/best.pt")  # Load a custom model.
	#print("Model info:")
	#print(model.info())

	# Train the model.
	model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)

	# Validate the model.
	metrics = model.val()  # No arguments needed, dataset and settings remembered.
	#	metrics.box.map  # mAP50-95(B).
	#	metrics.box.map50  # mAP50(B).
	#	metrics.box.map75  # mAP75(B).
	#	metrics.box.maps  # A list contains mAP50-95(B) of each category.
	#	metrics.seg.map  # mAP50-95(M).
	#	metrics.seg.map50  # mAP50(M).
	#	metrics.seg.map75  # mAP75(M).
	#	metrics.seg.maps  # A list contains mAP50-95(M) of each category.

	# Predict with the model.
	results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image.

	# Export the model.
	success = model.export(format="onnx")

# REF [site] >> https://docs.ultralytics.com/tasks/classification/
def yolov8_classification_example():
	# Load a model.
	if False:
		model = YOLO("yolov8n-cls.yaml")  # Build a new model from scratch.
	elif True:
		model = YOLO("yolov8n-cls.pt")  # Load a pretrained model (recommended for training).
	else:
		model = YOLO("/path/to/best.pt")  # Load a custom model.
	#print("Model info:")
	#print(model.info())

	# Train the model.
	model.train(data="mnist160", epochs=100, imgsz=64)

	# Validate the model.
	metrics = model.val()  # No arguments needed, dataset and settings remembered.
	#	metrics.top1  # Top1 accuracy.
	#	metrics.top5  # Top5 accuracy.

	# Predict with the model.
	results = model("https://ultralytics.com/images/bus.jpg")  # Predict on an image.

	# Export the model
	success = model.export(format="onnx")

def main():
	# Model:
	#	https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models
	#
	#	YOLOv8:
	#		Detection: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x.
	#		Instance segmentation: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg.
	#		Classification: yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls.
	#	YOLOv5:
	#		Detection: yolov5nu, yolov5su, yolov5mu, yolov5lu, yolov5xu.

	# Detection.
	yolov8_detection_example()
	#yolov5_detection_example()

	# Instance segmentation.
	#yolov8_segmentation_example()

	# Classification.
	#yolov8_classification_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
