#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import cv2
import tensorflow as tf
from nms import nms
# REF [site] >>
#	https://github.com/cudamat/cudamat/blob/6565e63a23a2d61b046b8d115346130da05e7d31/setup.py
#	https://github.com/MrGF/py-faster-rcnn-windows/blob/master/lib/setup.py
import py_cpu_nms, cpu_nms, gpu_nms
import py_parallel_nms_cpu, py_parallel_nms_gpu

def main():
	np.random.seed(37)

	max_x, max_y = 10000, 10000
	num_boxes = 100000

	#iou_threshold, score_threshold = 0.5, float('-inf')
	iou_threshold, score_threshold = 0.5, 0.001
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
	#boxes2 = np.zeros((num_boxes,), dtype=py_parallel_nms_cpu.PyBox)
	#for idx in range(num_boxes):
	#	boxes2[idx] = py_parallel_nms_cpu.PyBox(*boxes_scores[idx,:])
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('End generating data.')

	"""
	#--------------------
	# REF [site] >> https://github.com/rbgirshick/py-faster-rcnn
	#	Too slow.
	print('Start py_cpu_nms...')
	start_time = time.time()
	selected_indices = py_cpu_nms.py_cpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End py_cpu_nms.')

	#--------------------
	# REF [site] >> https://github.com/rbgirshick/py-faster-rcnn
	#	Too slow.
	print('Start cpu_nms...')
	start_time = time.time()
	selected_indices = cpu_nms.cpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End cpu_nms.')
	"""

	#--------------------
	# REF [site] >> https://github.com/rbgirshick/py-faster-rcnn
	print('Start gpu_nms...')
	start_time = time.time()
	selected_indices = gpu_nms.gpu_nms(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices)
	print('End gpu_nms.')

	"""
	#--------------------
	# REF [site] >> https://github.com/jeetkanjani7/Parallel_NMS
	#	Not good implementation.
	#	Too slow.
	print('Start parallel_nms_cpu...')
	start_time = time.time()
	is_kept_list = py_parallel_nms_cpu.py_parallel_nms_cpu(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', np.count_nonzero(is_kept_list))
	#print('\tSelected indices =', np.nonzero(is_kept_list))
	print('End parallel_nms_cpu.')

	#--------------------
	# REF [site] >> https://github.com/jeetkanjani7/Parallel_NMS
	#	Not good implementation.
	print('Start parallel_nms_gpu...')
	start_time = time.time()
	is_kept_list = py_parallel_nms_gpu.py_parallel_nms_gpu(boxes_scores, iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', np.count_nonzero(is_kept_list))
	#print('\tSelected indices =', np.nonzero(is_kept_list))
	print('End parallel_nms_gpu.')
	"""

	#--------------------
	# REF [site] >>
	#	https://bitbucket.org/tomhoag/nms
	#	https://nms.readthedocs.io/en/latest/index.html
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
	# REF [site] >> https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
	print('Start opencv...')
	start_time = time.time()
	selected_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores, score_threshold=score_threshold, nms_threshold=iou_threshold)
	print('\tElapsed time = {}'.format(time.time() - start_time))
	print('\t#selected boxes =', len(selected_indices))
	#print('\tSelected indices =', selected_indices.ravel().tolist())
	print('End opencv.')

	#--------------------
	# REF [site] >> https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
	#	Slow, not as expected.
	print('Start tf...')
	with tf.Session() as sess:
		# For CUDA context initialization.
		#A = tf.constant([1])
		#A.eval(session=sess)
		#sess.run(tf.global_variables_initializer())

		tf_nms = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold=iou_threshold, score_threshold=score_threshold)
		start_time = time.time()
		selected_indices = tf_nms.eval(session=sess)
		print('\tElapsed time = {}'.format(time.time() - start_time))
		print('\t#selected boxes =', len(selected_indices))
		#print('\tSelected indices =', selected_indices)
	print('End tf.')

#--------------------------------------------------------------------

# Usage:
#	REF [site] >> http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html
#	python setup_nms.py build_ext --inplace
#	python nms_test.py

if '__main__' == __name__:
	main()
