#!/usr/bin/env python

import time
import numpy as np
import cv2
#import tensorflow as tf
from nms import nms
# REF [site] >>
#	https://github.com/cudamat/cudamat/blob/6565e63a23a2d61b046b8d115346130da05e7d31/setup.py
#	https://github.com/MrGF/py-faster-rcnn-windows/blob/master/lib/setup.py
import py_cpu_nms, cpu_nms, gpu_nms

def main():
	np.random.seed(37)

	max_x, max_y = 10000, 10000
	num_boxes = 100000

	#iou_threshold, score_threshold = 0.5, float('-inf')
	iou_threshold, score_threshold = 0.8, 0.001
	max_output_size = num_boxes

	#--------------------
	print('Start generating data...')
	start_time = time.time()
	boxes = list()
	for _ in range(num_boxes):
		xs = np.random.randint(max_x, size=2)
		ys = np.random.randint(max_y, size=2)
		boxes.append([np.min(xs), np.min(ys), np.max(xs), np.max(ys)])
	boxes = np.array(boxes)
	scores = np.random.rand(num_boxes)
	boxes_scores = np.hstack([boxes, scores.reshape([-1, 1])]).astype(np.float32)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End generating data.')

	"""
	#--------------------
	print('Start py_cpu_nms...')
	start_time = time.time()
	selected_indices = py_cpu_nms.py_cpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End py_cpu_nms.')

	#--------------------
	print('Start cpu_nms...')
	start_time = time.time()
	selected_indices = cpu_nms.cpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End cpu_nms.')
	"""

	#--------------------
	print('Start gpu_nms...')
	start_time = time.time()
	selected_indices = gpu_nms.gpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End gpu_nms.')

	#--------------------
	print('Start nms...')
	nms_function = nms.fast.nms
	#nms_function = nms.felzenszwalb.nms
	#nms_function = nms.malisiewicz.nms
	start_time = time.time()
	selected_indices = nms.boxes(boxes, scores, score_threshold=score_threshold, nms_threshold=iou_threshold, nms_function=nms_function)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End nms.')

	#--------------------
	print('Start opencv...')
	start_time = time.time()
	selected_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores, score_threshold=score_threshold, nms_threshold=iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices.ravel().tolist())
	print('End opencv.')

	"""
	#--------------------
	print('Start tf...')
	sess = tf.Session()
	tf_nms = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=iou_threshold, score_threshold=score_threshold)
	start_time = time.time()
	selected_indices = tf_nms.eval(session=sess)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End tf.')
	"""

#%%------------------------------------------------------------------

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup_nms.py build_ext --inplace
#	python nms_test.py

if '__main__' == __name__:
	main()
