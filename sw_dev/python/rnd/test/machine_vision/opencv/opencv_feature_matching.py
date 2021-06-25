#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2 as cv

# REF [site] >> https://docs.opencv.org/4.1.0/d7/dff/tutorial_feature_homography.html
def simple_feature_matching_example():
	object_image_filepath = '../../../data/machine_vision/box.png'
	scene_image_filepath = '../../../data/machine_vision/box_in_scene.png'
	
	img_object = cv.imread(object_image_filepath, cv.IMREAD_GRAYSCALE)
	if img_object is None:
		print('Failed to load an object image, {}.'.format(object_image_filepath))
		return
	img_scene = cv.imread(scene_image_filepath, cv.IMREAD_GRAYSCALE)
	if img_scene is None:
		print('Failed to load a scene image, {}.'.format(scene_image_filepath))
		return

	# Detect the keypoints and compute the descriptors.
	if False:
		hessian_threshold = 400
		detector = cv.xfeatures2d_SURF.create(hessianThreshold=hessian_threshold)
		norm = cv.NORM_L2
	elif False:
		detector = cv.xfeatures2d_SIFT.create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
		norm = cv.NORM_L2
	elif True:
		num_features = 500
		detector = cv.ORB.create(nfeatures=num_features, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
		norm = cv.NORM_HAMMING
	elif False:
		detector = cv.AKAZE.create(descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB, descriptor_size=0, descriptor_channels=3, threshold=0.001, nOctaves=4, nOctaveLayers=4, diffusivity=cv.KAZE_DIFF_PM_G2)
		norm = cv.NORM_HAMMING
	elif False:
		detector = cv.BRISK.create(thresh=30, octaves=3, patternScale=1.0)
		norm = cv.NORM_HAMMING

	keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
	keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

	# Match descriptor vectors.
	if True:
		# REF [file] >> ${OPENCV_HOME}/modules/flann/include/opencv2/flann/defines.h
		FLANN_INDEX_LINEAR = 0
		FLANN_INDEX_KDTREE = 1
		FLANN_INDEX_KMEANS = 2
		FLANN_INDEX_COMPOSITE = 3
		FLANN_INDEX_KDTREE_SINGLE = 4
		FLANN_INDEX_HIERARCHICAL = 5
		FLANN_INDEX_LSH = 6
		FLANN_INDEX_SAVED = 254
		FLANN_INDEX_AUTOTUNED = 255

		#matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
		if norm == cv.NORM_L2:
			matcher = cv.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), {})
		else:
			#matcher = cv.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2), {})
			matcher = cv.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1), {})
	elif False:
		matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	elif False:
		matcher = cv.BFMatcher(norm)

	if False:
		matches = matcher.match(descriptors_obj, descriptors_scene)

		# Sort matches by score.
		matches.sort(key=lambda x: x.distance, reverse=False)

		# Remove not so good matches.
		GOOD_MATCH_PERCENT = 0.15
		num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
		good_matches = matches[:num_good_matches]
	else:
		matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

		# Filter matches using the Lowe's ratio test.
		ratio_thresh = 0.75
		good_matches = list()
		for mth in matches:
			if len(mth) == 2:
				m, n = mth
				if m.distance < ratio_thresh * n.distance:
					good_matches.append(m)

	# Draw matches.
	img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
	cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

	# Localize the object.
	obj = np.empty((len(good_matches),2), dtype=np.float32)
	scene = np.empty((len(good_matches),2), dtype=np.float32)
	for i in range(len(good_matches)):
		# Get the keypoints from the good matches.
		obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
		obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
		scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
		scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
	H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

	# Get the corners from the image_1 (the object to be 'detected').
	obj_corners = np.empty((4, 1, 2), dtype=np.float32)
	obj_corners[0,0,0] = 0
	obj_corners[0,0,1] = 0
	obj_corners[1,0,0] = img_object.shape[1]
	obj_corners[1,0,1] = 0
	obj_corners[2,0,0] = img_object.shape[1]
	obj_corners[2,0,1] = img_object.shape[0]
	obj_corners[3,0,0] = 0
	obj_corners[3,0,1] = img_object.shape[0]

	scene_corners = cv.perspectiveTransform(obj_corners, H)

	# Draw lines between the corners (the mapped object in the scene - image_2).
	cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
		(int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
	cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
		(int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
	cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
		(int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
	cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
		(int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

	# Show detected matches.
	cv.imshow('Good Matches & Object detection', img_matches)
	cv.waitKey()

def main():
	simple_feature_matching_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
