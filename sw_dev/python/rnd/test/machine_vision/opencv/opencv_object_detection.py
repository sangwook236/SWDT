#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2 as cv

# REF [site] >>
#	https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
#	https://github.com/tensorflow/models/tree/master/research/object_detection
#	https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
def tensorflow_object_detection_example():
	cvNet = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

	img = cv.imread('example.jpg')
	rows = img.shape[0]
	cols = img.shape[1]
	cvNet.setInput(cv.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
	cvOut = cvNet.forward()

	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.3:
			left = detection[3] * cols
			top = detection[4] * rows
			right = detection[5] * cols
			bottom = detection[6] * rows
			cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)

	cv.imshow('img', img)
	cv.waitKey()

def main():
	tensorflow_object_detection_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
