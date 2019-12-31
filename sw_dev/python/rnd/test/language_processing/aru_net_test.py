#!/usr/bin/env python

from __future__ import print_function, division

import os, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imageio
from PIL import Image
from pix_lab.util.util import load_graph, read_image_list
#from pix_lab.util.inference_pb import Inference_pb
import cv2

def load_image(path, scale, mode):
	aImg = imageio.imread(path, pilmode=mode)
	sImg = np.array(Image.fromarray(aImg).resize((round(aImg.shape[1] * scale), round(aImg.shape[0] * scale)), resample=Image.BICUBIC))
	fImg = sImg
	if len(sImg.shape) == 2:
		fImg = np.expand_dims(fImg, 2)
	fImg = np.expand_dims(fImg, 0)

	return fImg

# REF [file] >> ${ARU-Net_HOME}/pix_lab/util/inference_pb.py
def inference(path_to_pb, img_list, scale=1.0, mode='L', print_result=True, gpu_device='0'):
	graph = load_graph(path_to_pb)
	val_size = len(img_list)
	if val_size is None:
		print('No Inference Data available. Skip Inference.')
		return

	output_dir_path = './prediction'
	os.makedirs(output_dir_path, exist_ok=False)

	session_conf = tf.ConfigProto()
	session_conf.gpu_options.visible_device_list = gpu_device
	with tf.Session(graph=graph, config=session_conf) as sess:
		x = graph.get_tensor_by_name('inImg:0')
		predictor = graph.get_tensor_by_name('output:0')
		print('Start Inference...')
		timeSum = 0.0
		for step in range(0, val_size):
			aTime = time.time()
			aImgPath = img_list[step]
			print('Image: {:} '.format(aImgPath))
			batch_x = load_image(aImgPath, scale, mode)
			print('Resolution: h {:}, w {:} '.format(batch_x.shape[1],batch_x.shape[2]))

			# Run validation.
			aPred = sess.run(predictor, feed_dict={x: batch_x})

			curTime = (time.time() - aTime) * 1000.0
			timeSum += curTime
			print('Update time: {:.2f} ms'.format(curTime))

			if print_result:
				n_class = aPred.shape[3]
				channels = batch_x.shape[3]
				"""
				fig = plt.figure()
				for aI in range(0, n_class+1):
					if aI == 0:
						a = fig.add_subplot(1, n_class+1, 1)
						if channels == 1:
							plt.imshow(batch_x[0, :, :, 0], cmap=plt.cm.gray)
						else:
							plt.imshow(batch_x[0, :, :, :])
						a.set_title('input')
					else:
						a = fig.add_subplot(1, n_class+1, aI+1)
						plt.imshow(aPred[0,:, :,aI-1], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
						#misc.imsave('out' + str(aI) + '.jpg', aPred[0,:, :,aI-1])
						a.set_title('Channel: ' + str(aI-1))
				print('To go on just CLOSE the current plot.')
				plt.show()
				"""
				"""
				for cls in range(0, n_class):
					print('***', np.min(aPred[0,:, :,cls]), np.max(aPred[0,:, :,cls]))
					pred = aPred[0,:, :,cls]
					if cls < 2:
						pred[pred > 0.5] = 1
					else:
						pred[pred < 0.5] = 0.0
					cv2.imshow('Class ' + str(cls), pred)
				"""
				if 1 == channels:
					rgb = cv2.cvtColor(batch_x[0, :, :, 0], cv2.COLOR_GRAY2BGR)
				else:
					rgb = batch_x[0, :, :, :]
				cls = 0
				pred = aPred[0,:, :,cls]
				if 0 == cls:
					rgb[pred > 0.1] = (0, 0, 255)
				elif 1 == cls:
					rgb[pred > 0.2] = (0, 0, 255)
				else:
					rgb[pred < 0.5] = (0, 0, 255)
				cv2.imwrite(os.path.join(output_dir_path, 'prediction{}_{}.tif'.format(cls, step)), rgb)
				#cv2.imshow('Prediction', pred)
				#cv2.imshow('Overlay', rgb)
				#cv2.waitKey(0)

		print('Inference avg update time: {:.2f} ms'.format(timeSum / val_size))
		print('Inference Finished!')

def main():
	if 'posix' == os.name:
		aru_net_dir_path = '/home/sangwook/lib_repo/python/ARU-Net_github'
	else:
		aru_net_dir_path = 'D:/lib_repo/python/rnd/ARU-Net_github'

	path_to_pb = os.path.join(aru_net_dir_path, 'demo_nets/model100_ema.pb')  # ${ARU-Net_HOME}/demo_nets/model100_ema.pb
	#path_list_imgs = os.path.join(aru_net_dir_path, 'demo_images/imgs.lst')  # ${ARU-Net_HOME}/demo_images/imgs.lst
	path_list_imgs = './epapyrus_images.lst'
	#path_list_imgs = './keit_images.lst'

	img_list = read_image_list(path_list_imgs)
	inference(path_to_pb, img_list, scale=1.0, mode='L', print_result=True, gpu_device='0')

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
