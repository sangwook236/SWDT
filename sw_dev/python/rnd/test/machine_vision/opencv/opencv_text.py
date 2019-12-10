#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://docs.opencv.org/4.1.0/da/d56/group__text__detect.html
#	https://docs.opencv.org/4.1.0/d8/df2/group__text__recognize.html

import math
import numpy as np
import cv2 as cv

# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/detect_er_chars.py
def ER_char_detector_example():
	image_filepaths = [
		'./image.png',
	]

	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/trained_classifierNM1.xml
	trained_classifierNM1_filepath = './trained_classifierNM1.xml'
	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/trained_classifierNM1.xml
	trained_classifierNM2_filepath = './trained_classifierNM2.xml'

	# Reads an Extremal Region Filter for the 1st stage classifier of N&M algorithm from the provided path.
	erc1 = cv.text.loadClassifierNM1(trained_classifierNM1_filepath)
	er1 = cv.text.createERFilterNM1(erc1)

	# Reads an Extremal Region Filter for the 2nd stage classifier of N&M algorithm from the provided path.
	erc2 = cv.text.loadClassifierNM2(trained_classifierNM2_filepath)
	er2 = cv.text.createERFilterNM2(erc2)

	for image_filepath in image_filepaths:
		img = cv.imread(image_filepath, cv.IMREAD_COLOR)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue
		gray = cv.imread(image_filepath, cv.IMREAD_GRAYSCALE)
		if gray is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue

		regions = cv.text.detectRegions(gray, er1, er2)

		# Visualization.
		rects = [cv.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
		for rect in rects:
			cv.rectangle(img, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
		for rect in rects:
			cv.rectangle(img, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)

		cv.imshow('Text detection result', img)
		cv.waitKey(0)

	cv.destroyAllWindows()

# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/textdetection.py
def ER_text_detector_example():
	image_filepaths = [
		'./image.png',
	]

	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/trained_classifierNM1.xml
	trained_classifierNM1_filepath = './trained_classifierNM1.xml'
	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/trained_classifierNM1.xml
	trained_classifierNM2_filepath = './trained_classifierNM2.xml'
	# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/trained_classifier_erGrouping.xml
	trained_classifier_erGrouping_filepath = './trained_classifier_erGrouping.xml'

	# Reads an Extremal Region Filter for the 1st stage classifier of N&M algorithm from the provided path.
	erc1 = cv.text.loadClassifierNM1(trained_classifierNM1_filepath)
	er1 = cv.text.createERFilterNM1(erc1, 16, 0.00015, 0.13, 0.2, True, 0.1)

	# Reads an Extremal Region Filter for the 2nd stage classifier of N&M algorithm from the provided path.
	erc2 = cv.text.loadClassifierNM2(trained_classifierNM2_filepath)
	er2 = cv.text.createERFilterNM2(erc2, 0.5)

	for image_filepath in image_filepaths:
		img = cv.imread(image_filepath, cv.IMREAD_COLOR)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue

		# Extract channels to be processed individually.
		channels = cv.text.computeNMChannels(img)

		# Append negative channels to detect ER- (bright regions over dark background).
		cn = len(channels) - 1
		for c in range(0, cn):
			channels.append((255 - channels[c]))

		# Apply the default cascade classifier to each independent channel (could be done in parallel).
		print('Extracting Class Specific Extremal Regions from ' + str(len(channels)) + ' channels ...')
		print('    (...) this may take a while (...)')
		vis = img.copy()
		for channel in channels:
			regions = cv.text.detectRegions(channel, er1, er2)

			rects = cv.text.erGrouping(img, channel, [r.tolist() for r in regions])
			#rects = cv.text.erGrouping(img, channel, [x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY, trained_classifier_erGrouping_filepath, 0.5)

			# Visualization.
			for r in range(0, np.shape(rects)[0]):
				rect = rects[r]
				cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
				cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)

		# Visualization.
		cv.imshow('Text detection result', vis)
		cv.waitKey(0)

	cv.destroyAllWindows()

def decode(scores, geometry, scoreThresh):
	############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
	assert len(scores.shape) == 4, 'Incorrect dimensions of scores'
	assert len(geometry.shape) == 4, 'Incorrect dimensions of geometry'
	assert scores.shape[0] == 1, 'Invalid dimensions of scores'
	assert geometry.shape[0] == 1, 'Invalid dimensions of geometry'
	assert scores.shape[1] == 1, 'Invalid dimensions of scores'
	assert geometry.shape[1] == 5, 'Invalid dimensions of geometry'
	assert scores.shape[2] == geometry.shape[2], 'Invalid dimensions of scores and geometry'
	assert scores.shape[3] == geometry.shape[3], 'Invalid dimensions of scores and geometry'

	detections = []
	confidences = []

	height = scores.shape[2]
	width = scores.shape[3]
	for y in range(0, height):
		# Extract data from scores.
		scoresData = scores[0][0][y]
		x0_data = geometry[0][0][y]
		x1_data = geometry[0][1][y]
		x2_data = geometry[0][2][y]
		x3_data = geometry[0][3][y]
		anglesData = geometry[0][4][y]
		for x in range(0, width):
			score = scoresData[x]

			# If score is lower than threshold score, move to next x.
			if(score < scoreThresh):
				continue

			# Calculate offset.
			offsetX = x * 4.0
			offsetY = y * 4.0
			angle = anglesData[x]

			# Calculate cos and sin of angle.
			cosA = math.cos(angle)
			sinA = math.sin(angle)
			h = x0_data[x] + x2_data[x]
			w = x1_data[x] + x3_data[x]

			# Calculate offset.
			offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

			# Find points for rectangle.
			p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
			p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
			center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
			detections.append((center, (w, h), -angle * 180.0 / math.pi))
			confidences.append(float(score))

	# Return detections and confidences.
	return [detections, confidences]

# REF [file] >> ${OPENCV_HOME}/samples/dnn/text_detection.py
def EAST_detector_example():
	image_filepaths = [
		'./image.png',
	]

	confThreshold = 0.5  # Confidence threshold.
	nmsThreshold = 0.4  # Non-maximum suppression threshold.
	inpWidth = 320
	inpHeight = 320

	# Path to a binary .pb file of model contains trained weights.
	# REF [site] >> https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/download_models.py
	model_filepath = './frozen_east_text_detection.pb'

	# Load network.
	net = cv.dnn.readNet(model_filepath)

	# Create a new named window.
	kWinName = 'EAST: An Efficient and Accurate Scene Text Detector'
	cv.namedWindow(kWinName, cv.WINDOW_NORMAL)

	outNames = []
	outNames.append('feature_fusion/Conv_7/Sigmoid')
	outNames.append('feature_fusion/concat_3')

	for image_filepath in image_filepaths:
		img = cv.imread(image_filepath, cv.IMREAD_COLOR)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue

		# Get image height and width.
		height_ = img.shape[0]
		width_ = img.shape[1]
		rW = width_ / float(inpWidth)
		rH = height_ / float(inpHeight)

		# Create a 4D blob from image.
		blob = cv.dnn.blobFromImage(img, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

		# Run the model.
		net.setInput(blob)
		outs = net.forward(outNames)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

		# Get scores and geometry.
		scores = outs[0]
		geometry = outs[1]
		[boxes, confidences] = decode(scores, geometry, confThreshold)

		# Apply NMS.
		indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
		for i in indices:
			# Get 4 corners of the rotated rect.
			vertices = cv.boxPoints(boxes[i[0]])
			# Scale the bounding box coordinates based on the respective ratios.
			for j in range(4):
				vertices[j][0] *= rW
				vertices[j][1] *= rH
			for j in range(4):
				p1 = (vertices[j][0], vertices[j][1])
				p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
				cv.line(img, p1, p2, (0, 255, 0), 1);

		# Put efficiency information.
		cv.putText(img, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

		# Display the image.
		cv.imshow(kWinName, img)
		cv.waitKey(0)

	cv.destroyAllWindows()

# REF [file] >>
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/deeptextdetection.py
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/textbox_demo.cpp
def TextBoxes_detector_example():
	image_filepaths = [
		'./image.png',
	]

	if True:
		# REF [site] >> https://github.com/MhLiao/TextBoxes

		# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/textbox.prototxt
		# REF [file] >> ${TextBoxes_HOME}/examples/TextBoxes/deploy.prototxt
		textbox_prototxt_filepath = './TextBox.prototxt'
		# REF [site] >> https://www.dropbox.com/s/g8pjzv2de9gty8g/TextBoxes_icdar13.caffemodel?dl=0
		textbox_caffemodel_filepath = './TextBoxes_icdar13.caffemodel'
	else:
		# Error:
		#	ReadProtoFromTextFile(param_file, param). Failed to parse NetParameter file: ./TextBoxespp_deploy.prototxt in function cv::dnn::ReadNetParamsFromTextFileOrDie

		# REF [site] >> https://github.com/MhLiao/TextBoxes_plusplus
	
		# REF [file] >> ${TextBoxes_plusplus_HOME}/models/deploy.prototxt
		textbox_prototxt_filepath = './TextBoxespp_deploy.prototxt'
		# REF [site] >> https://www.dropbox.com/s/kpv17f3syio95vn/model_pre_train_syn.caffemodel?dl=0
		#textbox_caffemodel_filepath = './TextBoxespp_pre_train_syn.caffemodel'
		# REF [site] >> https://www.dropbox.com/s/9znpiqpah8rir9c/model_icdar15.caffemodel?dl=0
		textbox_caffemodel_filepath = './TextBoxespp_icdar15.caffemodel'

	textSpotter = cv.text.TextDetectorCNN_create(textbox_prototxt_filepath, textbox_caffemodel_filepath)
	for image_filepath in image_filepaths:
		img = cv.imread(image_filepath, cv.IMREAD_COLOR)
		if img is None:
			print('Failed to load an image, {}.'.format(image_filepath))
			continue

		rects, outProbs = textSpotter.detect(img)

		vis = img.copy()
		thres = 0.6
		for r in range(np.shape(rects)[0]):
			if outProbs[r] > thres:
				rect = rects[r]
				cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)

		cv.imshow('Text detection result', vis)
		cv.waitKey(0)

	cv.destroyAllWindows()

# REF [file] >>
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/character_recognition.cpp
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/segmented_word_recognition.cpp
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/webcam_demo.cpp
def HMM_decoder_example():
	raise NotImplementedError

# REF [file] >> ${OPENCV_CONTRIB_HOME}/modules/text/samples/cropped_word_recognition.cpp
def beam_search_decoder_example():
	raise NotImplementedError

# REF [file] >>
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/end_to_end_recognition.cpp
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/segmented_word_recognition.cpp
#	${OPENCV_CONTRIB_HOME}/modules/text/samples/webcam_demo.cpp
def tesseract_example():
	raise NotImplementedError

def main():
	#ER_char_detector_example()
	#ER_text_detector_example()
	#EAST_detector_example()
	TextBoxes_detector_example()

	#HMM_decoder_example()  # Not yet implemented.
	#beam_search_decoder_example()  # Not yet implemented.
	#tesseract_example()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
