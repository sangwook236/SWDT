#!/usr/bin/env python

# REF [site] >> https://github.com/lucasb-eyer/pydensecrf

import time
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# REF [site] >> https://github.com/lucasb-eyer/pydensecrf/blob/master/examples/inference.py
def simple_example():
	image_filepath = './driver_license.png'
	#image_filepath = './driver_license_20190329.jpg'
	#image_filepath = './passport_chaewanlee_20130402.jpg'
	#image_filepath = './passport_chaewanlee_20170804.jpg'
	#image_filepath = './passport_hyejoongkim_20140508.jpg'
	#image_filepath = './passport_jihyunglee_20130402.jpg'
	#image_filepath = './passport_jihyunglee_20170804.jpg'
	#image_filepath = './passport_malnamkang_1.jpg'
	#image_filepath = './passport_malnamkang_2.jpg'
	#image_filepath = './passport_sangwooklee_20031211.jpg'
	#image_filepath = './passport_sangwooklee_20130402.jpg'
	#image_filepath = './rrn_malnamkang.jpg
	#image_filepath = './rrn_sangwooklee_20190329.jpg'
	anno_filepath = image_filepath

	img = cv2.imread(image_filepath)

	# Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR.
	anno_rgb = cv2.imread(anno_filepath).astype(np.uint32)
	anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

	# Convert the 32bit integer color to 1, 2, ..., labels.
	# Note that all-black, i.e. the value 0 for background will stay 0.
	colors, labels = np.unique(anno_lbl, return_inverse=True)

	# But remove the all-0 black, that won't exist in the MAP!
	HAS_UNK = 0 in colors
	if HAS_UNK:
		print('Found a full-black pixel in annotation image, assuming it means "unknown" label, and will thus not be present in the output!')
		print('If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.')
		colors = colors[1:]
	#else:
	#	print('No single full-black pixel found in annotation image. Assuming there\'s no "unknown" label!')

	# And create a mapping back from the labels to 32bit integer colors.
	colorize = np.empty((len(colors), 3), np.uint8)
	colorize[:,0] = (colors & 0x0000FF)
	colorize[:,1] = (colors & 0x00FF00) >> 8
	colorize[:,2] = (colors & 0xFF0000) >> 16

	# Compute the number of classes in the label image.
	# We subtract one because the number shouldn't include the value 0 which stands for "unknown" or "unsure".
	n_labels = len(set(labels.flat)) - int(HAS_UNK)
	#print(n_labels, 'labels', ('plus "unknown" 0: ' if HAS_UNK else ''), set(labels.flat))
	print(n_labels, 'labels', ('plus "unknown" 0: ' if HAS_UNK else ''))

	#--------------------
	# Setup the CRF model.

	use_2d = False
	#use_2d = True

	print('Start building a CRF model...')
	start_time = time.time()
	if use_2d:
		print('Using 2D specialized functions.')

		# Example using the DenseCRF2D code.
		d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

		# Get unary potentials (neg log probability).
		U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
		d.setUnaryEnergy(U)

		# This adds the color-independent term, features are the locations only.
		d.addPairwiseGaussian(sxy=(3, 3), compat=3, 
							  kernel=dcrf.DIAG_KERNEL,
							  normalization=dcrf.NORMALIZE_SYMMETRIC)

		# This adds the color-dependent term, i.e. features are (x, y, r, g, b).
		d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10,
							   kernel=dcrf.DIAG_KERNEL,
							   normalization=dcrf.NORMALIZE_SYMMETRIC)
	else:
		print('Using generic 2D functions.')

		# Example using the DenseCRF class and the util functions.
		d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

		# Get unary potentials (neg log probability).
		U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
		d.setUnaryEnergy(U)

		# This creates the color-independent features and then add them to the CRF.
		feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
		d.addPairwiseEnergy(feats, compat=3,
							kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)

		# This creates the color-dependent features and then add them to the CRF.
		feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13), img=img, chdim=2)
		d.addPairwiseEnergy(feats, compat=10,
							kernel=dcrf.DIAG_KERNEL,
							normalization=dcrf.NORMALIZE_SYMMETRIC)
	print('End building a CRF model: {} secs.'.format(time.time() - start_time))

	#--------------------
	# Do inference and compute MAP.

	print('Start inferring by the CRF model...')
	start_time = time.time()
	# Run five inference steps.
	Q = d.inference(5)

	# Find out the most probable class for each pixel.
	MAP = np.argmax(Q, axis=0)
	print('End inferring by the CRF model: {} secs.'.format(time.time() - start_time))

	# Convert the MAP (labels) back to the corresponding colors and save the image.
	# Note that there is no "unknown" here anymore, no matter what we had at first.
	MAP = colorize[MAP,:]
	MAP = MAP.reshape(img.shape)

	#cv2.imwrite(output_filepath, MAP)
	cv2.imshow('Inference result', MAP)
	cv2.waitKey(0)

	#--------------------
	# Just randomly manually run inference iterations.

	print('Start manually inferring by the CRF model...')
	start_time = time.time()
	Q, tmp1, tmp2 = d.startInference()
	for i in range(5):
		print('KL-divergence at {}: {}.'.format(i, d.klDivergence(Q)))
		d.stepInference(Q, tmp1, tmp2)
	print('End manually inferring by the CRF model: {} secs.'.format(time.time() - start_time))

def main():
	simple_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
