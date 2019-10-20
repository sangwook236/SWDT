#!/usr/bin/env python

# REF [site] >> https://github.com/keras-team/keras-applications

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras_applications

def resnet_test():
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# ResNet50, ResNet101, ResNet152.
	model = keras_applications.resnet.ResNet50(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000,
		**kwargs
	)
	print(model.summary())

	# ResNet50V2, ResNet101V2, ResNet152V2.
	model = keras_applications.resnet_v2.ResNet50V2(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000,
		**kwargs
	)
	print(model.summary())

def resnext_test():
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# ResNeXt50, ResNeXt101.
	model = keras_applications.resnext.ResNeXt50(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000,
		**kwargs
	)
	print(model.summary())

def densenet_test():
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# DenseNet121, DenseNet169, DenseNet201.
	model = keras_applications.densenet.DenseNet121(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000,
		**kwargs
	)
	print(model.summary())

def nasnet_test():
	kwargs = {'backend': tf.keras.backend, 'layers': tf.keras.layers, 'models': tf.keras.models, 'utils': tf.keras.utils}

	# NASNetLarge, NASNetMobile.
	model = keras_applications.nasnet.NASNetLarge(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000,
		**kwargs
	)
	print(model.summary())

def main():
	#resnet_test()
	#resnext_test()
	#densenet_test()
	nasnet_test()

	#inception_test()  # Not yet implemented.
	#inception_resnet_test()  # Not yet implemented.
	#xception_test()  # Not yet implemented.
	#mobilenet_test()  # Not yet implemented.
	#efficientnet_test()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
