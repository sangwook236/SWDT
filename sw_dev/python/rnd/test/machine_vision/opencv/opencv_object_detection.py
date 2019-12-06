#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import cv2 as cv

# REF [site] >> https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py
def object_detection_example():
	if 'posix' == os.name:
		model_base_dir_path = '/home/sangwook/util_portable/annotation_tool'
	else:
		model_base_dir_path = 'D:/util_portable/annotation_tool'
	model_dir_path = model_base_dir_path + '/pretrained_model/ssd_mobilenet_v1_coco_2018_01_28'
	#model_dir_path = model_base_dir_path + '/pretrained_model/faster_rcnn_resnet50_coco_2018_01_28'
	#model_dir_path = model_base_dir_path + '/pretrained_model/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

	"""
	model: Binary file contains trained weights.
		The following file extensions are expected for models from different frameworks:
			*.caffemodel (Caffe, http://caffe.berkeleyvision.org/)
			*.pb (TensorFlow, https://www.tensorflow.org/)
			*.t7 | *.net (Torch, http://torch.ch/)
			*.weights (Darknet, https://pjreddie.com/darknet/)
			*.bin (DLDT, https://software.intel.com/openvino-toolkit)
			*.onnx (ONNX, https://onnx.ai/)
	config: Text file contains network configuration.
		It could be a file with the following extensions:
			*.prototxt (Caffe, http://caffe.berkeleyvision.org/)
			*.pbtxt (TensorFlow, https://www.tensorflow.org/)
			*.cfg (Darknet, https://pjreddie.com/darknet/)
			*.xml (DLDT, https://software.intel.com/openvino-toolkit)
		REF [site] >> https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
	"""
	model_filepath = model_dir_path + '/...'  # A .pb file with weights.
	config_filepath = model_dir_path + '/...'  # A .pxtxt file which contains network configuration.

	image_filepath = './coco.jpg'
	image_width, image_height = 300, 300

	raise NotImplementedError

# REF [site] >>
#	https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
#	https://github.com/tensorflow/models/tree/master/research/object_detection
#	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/tensorflow/tensorflow_object_detection.py
def tensorflow_object_detection_example():
	if 'posix' == os.name:
		model_base_dir_path = '/home/sangwook/util_portable/annotation_tool'
	else:
		model_base_dir_path = 'D:/util_portable/annotation_tool'
	model_dir_path = model_base_dir_path + '/pretrained_model/ssd_mobilenet_v1_coco_2018_01_28'
	#model_dir_path = model_base_dir_path + '/pretrained_model/faster_rcnn_resnet50_coco_2018_01_28'
	#model_dir_path = model_base_dir_path + '/pretrained_model/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

	model_filepath = model_dir_path + '/frozen_inference_graph.pb'  # A .pb file with weights.
	# Generate a config file:
	#	cd {OPENCV_HOME}/samples/dnn
	#	python tf_text_graph_ssd.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt
	#	python tf_text_graph_faster_rcnn.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt
	#	python tf_text_graph_mask_rcnn.py --input /path/to/model.pb --config /path/to/example.config --output /path/to/graph.pbtxt
	# REF [site] >>
	#	https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
	#	https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
	config_filepath = model_dir_path + '/graph.pbtxt'  # A .pxtxt file which contains network configuration.

	image_filepath = './coco.jpg'
	image_width, image_height = 300, 300

	img = cv.imread(image_filepath)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return
	rows, cols = img.shape[:2]

	cvNet = cv.dnn.readNetFromTensorflow(model_filepath, config_filepath)
	cvNet.setInput(cv.dnn.blobFromImage(img, size=(image_width, image_height), swapRB=True, crop=False))
	cvOut = cvNet.forward()

	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.3:
			left = detection[3] * cols
			top = detection[4] * rows
			right = detection[5] * cols
			bottom = detection[6] * rows
			cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

	cv.imshow('Image', img)
	cv.waitKey()

def showLegend(classes, legend=None):
	if not classes is None and legend is None:
		blockHeight = 30
		assert(len(classes) == len(colors))

		legend = np.zeros((blockHeight * len(colors), 200, 3), np.uint8)
		for i in range(len(classes)):
			block = legend[i * blockHeight:(i + 1) * blockHeight]
			block[:,:] = colors[i]
			cv.putText(block, classes[i], (0, blockHeight/2), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

		cv.namedWindow('Legend', cv.WINDOW_NORMAL)
		cv.imshow('Legend', legend)
		classes = None

def drawBox(image, class_id, conf, left, top, right, bottom):
	# Draw a bounding box.
	cv.rectangle(image, (left, top), (right, bottom), (0, 255, 0))

	label = '%.2f' % conf

	# Print a label of class.
	if classes:
		assert(class_id < len(classes))
		label = '%s: %s' % (classes[class_id], label)

	labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
	top = max(top, labelSize[1])
	cv.rectangle(image, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
	cv.putText(image, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

# REF [site] >> https://github.com/opencv/opencv/blob/master/samples/dnn/mask_rcnn.py
# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/tensorflow/tensorflow_object_detection.py
def tensorflow_mask_rcnn_example():
	if 'posix' == os.name:
		model_base_dir_path = '/home/sangwook/util_portable/annotation_tool'
	else:
		model_base_dir_path = 'D:/util_portable/annotation_tool'
	model_dir_path = model_base_dir_path + '/pretrained_model/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

	model_filepath = model_dir_path + '/frozen_inference_graph.pb'  # A .pb file with weights.
	config_filepath = model_dir_path + '/graph.pbtxt'  # A .pxtxt file which contains network configuration.
	class_filepath = None  # A text file with names of classes.
	color_filepath = None  # A text file with colors for an every class. An every color is represented with three values from 0 to 255 in BGR channels order.

	confidence_threshold = 0.5  # Confidence threshold.
	#nms_threshold = 0.4  # Non-maximum suppression threshold.
	#framework = 'tensorflow'  # An origin framework of the model. Detect it automatically if it does not set. {'caffe', 'tensorflow', 'torch', 'darknet', 'dldt'}.
	#backend = cv.dnn.DNN_BACKEND_DEFAULT  # Computation backends. {cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV}.
	#target = cv.dnn.DNN_TARGET_CPU  # Computation devices. {cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD}.

	image_filepath = './coco.jpg'
	image_width, image_height = 300, 300

	img = cv.imread(image_filepath)
	if img is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return
	imgH, imgW = img.shape[:2]
	legend = None

	# Load names of classes.
	classes = None
	if class_filepath:
		with open(class_filepath, 'rt') as fd:
			classes = fd.read().rstrip('\n').split('\n')

	# Load colors.
	colors = None
	if color_filepath:
		with open(color_filepath, 'rt') as fd:
			colors = [np.array(color.split(' '), np.uint8) for color in fd.read().rstrip('\n').split('\n')]

	# Load a network.
	#net = cv.dnn.readNet(cv.samples.findFile(model_filepath), cv.samples.findFile(config_filepath))
	net = cv.dnn.readNet(model_filepath, config_filepath)
	net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

	# Create a 4D blob from an image.
	blob = cv.dnn.blobFromImage(img, size=(image_width, image_height), swapRB=True, crop=False)

	# Run a model.
	net.setInput(blob)

	boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

	numClasses = masks.shape[1]
	numDetections = boxes.shape[2]

	# Draw segmentation.
	if not colors:
		# Generate colors.
		colors = [np.array([0, 0, 0], np.uint8)]
		for i in range(1, numClasses + 1):
			colors.append((colors[i - 1] + np.random.randint(0, 256, [3], np.uint8)) / 2)
		del colors[0]

	boxesToDraw = []
	for i in range(numDetections):
		box = boxes[0, 0, i]
		mask = masks[i]
		score = box[2]
		if score > confidence_threshold:
			class_id = int(box[1])
			left = int(imgW * box[3])
			top = int(imgH * box[4])
			right = int(imgW * box[5])
			bottom = int(imgH * box[6])

			left = max(0, min(left, imgW - 1))
			top = max(0, min(top, imgH - 1))
			right = max(0, min(right, imgW - 1))
			bottom = max(0, min(bottom, imgH - 1))

			boxesToDraw.append([frame, class_id, score, left, top, right, bottom])

			classMask = mask[class_id]
			classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
			mask = (classMask > 0.5)

			roi = frame[top:bottom+1, left:right+1][mask]
			frame[top:bottom+1, left:right+1][mask] = (0.7 * colors[class_id] + 0.3 * roi).astype(np.uint8)

	for box in boxesToDraw:
		drawBox(*box)

	# Put efficiency information.
	t, _ = net.getPerfProfile()
	label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
	cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

	showLegend(classes, legend)

	cv.imshow('Mask-RCNN in OpenCV', frame)

# REF [site] >> https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
def yolo_object_detection_example():
	if 'posix' == os.name:
		darknet_home_dir_path = '/home/sangwook/lib_repo/cpp/darknet_github'
	else:
		darknet_home_dir_path = 'D:/lib_repo/cpp/rnd/darknet_github'
		#darknet_home_dir_path = 'D:/lib_repo/cpp/darknet_github'

	config_filepath = darknet_home_dir_path + '/cfg/yolov3.cfg'
	weight_filepath = darknet_home_dir_path + '/yolov3.weights'
	label_filepath = darknet_home_dir_path + '/data/coco.names'
	image_filepath = darknet_home_dir_path + '/data/dog.jpg'

	confidence_threshold = 0.5  # Minimum probability to filter weak detections.
	nms_threshold = 0.3  # Threshold when applying non-maxima suppression.

	#--------------------
	LABELS = open(label_filepath).read().strip().split('\n')

	# Initialize a list of colors to represent each possible class label.
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

	#--------------------
	print('Load YOLO from disk...')
	net = cv.dnn.readNetFromDarknet(config_filepath, weight_filepath)

	image = cv.imread(image_filepath)
	if image is None:
		print('Failed to load an image, {}.'.format(image_filepath))
		return
	(H, W) = image.shape[:2]

	# Construct a blob from the input image.
	#blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	blob = cv.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
	net.setInput(blob)

	#--------------------
	# Determine only the *output* layer names that we need from YOLO.
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	print('YOLO output layers =', layer_names)

	# Perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities.
	start = time.time()
	layerOutputs = net.forward(layer_names)
	end = time.time()

	# Show timing information on YOLO.
	print('YOLO took {:.6f} seconds.'.format(end - start))
	# (19 * 19 * 3 = 1083, 85), (38 * 38 * 3 = 4332, 85), (76 * 96 * 3 = 17328, 85).
	print("YOLO outputs's shapes =", layerOutputs[0].shape, layerOutputs[1].shape, layerOutputs[2].shape)

	#--------------------
	# Initialize our lists of detected bounding boxes, confidences, and class IDs, respectively.
	boxes, confidences, classIDs = list(), list(), list()

	# Loop over each of the layer outputs.
	for output in layerOutputs:
		# Loop over each of the detections.
		for detection in output:
			# Extract the class ID and confidence (i.e., probability) of the current object detection.
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
			if confidence > confidence_threshold:
				# Scale the bounding box coordinates back relative to the size of the image,
				# keeping in mind that YOLO actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype('int')

				# Use the center (x, y)-coordinates to derive the top and and left corner of the bounding box.
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# Update our list of bounding box coordinates, confidences, and class IDs.
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	#--------------------
	# Apply non-maxima suppression to suppress weak, overlapping bounding boxes.
	idxs = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

	#--------------------
	# Ensure at least one detection exists.
	if len(idxs) > 0:
		# Loop over the indexes we are keeping.
		for i in idxs.flatten():
			# Extract the bounding box coordinates.
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# Draw a bounding box rectangle and label on the image.
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])
			cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Write the output image.
	cv.imwrite('./yolo_outputs.png', image)

	# Show the output image.
	#cv.imshow('Image', image)
	#cv.waitKey(0)

	# Write the model.
	# NOTE [info] >> Only TensorFlow models support export to text file in function 'writeTextGraph'.
	#cv.dnn.writeTextGraph(weight_filepath, './yolo_model.txt')

def main():
	# Caffe, darknet, Model Optimizer, ONNX, TensorFlow, Torch.

	#object_detection_example()  # Not yet implemented.

	#tensorflow_object_detection_example()  # SSD, R-CNN, Mask R-CNN.
	tensorflow_mask_rcnn_example()

	#yolo_object_detection_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
