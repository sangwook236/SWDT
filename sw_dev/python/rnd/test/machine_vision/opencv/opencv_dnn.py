#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://docs.opencv.org/master/d6/d0f/group__dnn.html

import os, time
import numpy as np
import cv2

# REF [site] >> https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
def yolo_object_detection():
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
	net = cv2.dnn.readNetFromDarknet(config_filepath, weight_filepath)

	image = cv2.imread(image_filepath)
	(H, W) = image.shape[:2]

	# Construct a blob from the input image.
	#blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (608, 608), swapRB=True, crop=False)
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
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

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
			cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
			text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])
			cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	# Write the output image.
	cv2.imwrite('./yolo_outputs.png', image)

	# Show the output image.
	#cv2.imshow('Image', image)
	#cv2.waitKey(0)

	# Write the model.
	# NOTE [info] >> Only TensorFlow models support export to text file in function 'writeTextGraph'.
	#cv2.dnn.writeTextGraph(weight_filepath, './yolo_model.txt')

def main():
	# Caffe, darknet, Model Optimizer, ONNX, TensorFlow, Torch.
	yolo_object_detection()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
