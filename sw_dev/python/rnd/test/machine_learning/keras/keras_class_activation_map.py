#!/usr/bin/env python
# coding: UTF-8

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2D
import tensorflow.keras.applications.resnet50 as resnet
import cv2
import matplotlib.pyplot as plt

def load_img(fname, input_size, preprocess_fn):
	original_img = cv2.imread(fname)[:, :, ::-1]
	original_size = (original_img.shape[1], original_img.shape[0])
	img = cv2.resize(original_img, (input_size, input_size))
	imgs = np.expand_dims(preprocess_fn(img), axis=0)
	return imgs, original_img, original_size

def get_cam_model(model_class, num_classes, input_size=224, last_conv_layer='activation_49', pred_layer='fc1000'):
	model = model_class(input_shape=(input_size, input_size, 3))
	model.summary()

	final_params = model.get_layer(pred_layer).get_weights()
	final_params = (final_params[0].reshape(1, 1, -1, num_classes), final_params[1])

	last_conv_output = model.get_layer(last_conv_layer).output
	x = UpSampling2D(size=(32, 32), interpolation='bilinear')(last_conv_output)
	x = Conv2D(filters=num_classes, kernel_size=(1, 1), name='predictions_2')(x)

	cam_model = Model(inputs=model.input, outputs=[model.output, x])
	cam_model.get_layer('predictions_2').set_weights(final_params)
	return cam_model

def postprocess(preds, cams, top_k=1):
	idxes = np.argsort(preds[0])[-top_k:]
	class_activation_map = np.zeros_like(cams[0,:,:,0])
	for i in idxes:
		class_activation_map += cams[0,:,:,i]
	return class_activation_map

# REF [site] >> https://github.com/keras-team/keras/blob/master/examples/class_activation_maps.py
# REF [paper] >> "Learning Deep Features for Discriminative Localization", CVPR 2016.
def class_activation_map_example():
	# Set an appropriate image file.
	image_filepath = './image.png'

	# The following parameters can be changed to other models that use global average pooling
	# e.g.) InceptionResnetV2 / NASNetLarge.
	NETWORK_INPUT_SIZE = 224
	MODEL_CLASS = resnet.ResNet50
	PREPROCESS_FN = resnet.preprocess_input
	LAST_CONV_LAYER = 'activation_48' #'activation_49'
	PRED_LAYER = 'fc1000'

	#--------------------
	# Number of imagenet classes.
	N_CLASSES = 1000

	# Load image.
	imgs, original_img, original_size = load_img(image_filepath, input_size=NETWORK_INPUT_SIZE, preprocess_fn=PREPROCESS_FN)

	# Predict.
	model = get_cam_model(MODEL_CLASS, N_CLASSES, NETWORK_INPUT_SIZE, LAST_CONV_LAYER, PRED_LAYER)
	preds, cams = model.predict(imgs)

	# Post processing.
	class_activation_map = postprocess(preds, cams)

	# Plot image+cam to original size.
	plt.imshow(original_img, alpha=0.5)
	plt.imshow(cv2.resize(class_activation_map, original_size), cmap='jet', alpha=0.5)
	plt.show()

def main():
	class_activation_map_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
