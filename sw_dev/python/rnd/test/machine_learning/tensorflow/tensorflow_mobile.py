#!/usr/bin/env python

# Mobile:
#	https://www.tensorflow.org/mobile/
#	REF [file] >> tensorflow_usage_guide.txt

# Android:
#	https://www.tensorflow.org/mobile/android_build
#		https://www.tensorflow.org/mobile/android_build#build_the_demo_using_android_studio
#		https://www.tensorflow.org/mobile/android_build#adding_tensorflow_to_your_apps_using_android_studio
#			=> Add 'org.tensorflow:tensorflow-android:+' in module-level build.gradle section instead of app-level.
#	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
#		${TENSORFLOW_HOME}/tensorflow/examples/android
#	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android
# IOS:
#	https://www.tensorflow.org/mobile/ios_build
#	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/ios
#		${TENSORFLOW_HOME}/tensorflow/examples/ios

# Preparing models for mobile deployment:
#	https://www.tensorflow.org/mobile/prepare_models/
#	REF [file] >> tensorflow_graph_tool.py
# Train a model -> Freeze a graph -> Optimize a frozen graph -> Convert to TensorFlow Lite:
#	https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3

# TensorFlow Android Inference Interface:
#	${TENSORFLOW_HOME}/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java
#		Exists in tensorflow-android-1.10.0.aar.
#	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/android
#	https://medium.com/capital-one-developers/using-a-pre-trained-tensorflow-model-on-android-e747831a3d6
#	https://medium.com/capital-one-developers/using-a-pre-trained-tensorflow-model-on-android-part-2-153ebdd4c465
#	https://medium.com/joytunes/deploying-a-tensorflow-model-to-android-69d04d1b0cba

# TensorFlow Lite:
#	https://www.tensorflow.org/mobile/tflite/
#		${TENSORFLOW_HOME}/tensorflow/contrib/lite
#	https://www.tensorflow.org/mobile/tflite/demo_android
#	https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite
#	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/
#		${TENSORFLOW_HOME}/tensorflow/contrib/lite/java/demo
#	https://medium.com/tensorflow/using-tensorflow-lite-on-android-9bbc9cb7d69d
#	https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3
#
#	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/models.md

# Convert a model to tflite:
#	tflite_convert:
#		cd ${TENSORFLOW_HOME}
#		bazel build tensorflow/contrib/lite/python:tflite_convert
#		bazel-bin/tensorflow/contrib/lite/python/tflite_convert --help
#	python tflite_convert.py:
#		${TENSORFLOW_HOME}/tensorflow/contrib/lite/python/tflite_convert.py
#		${TENSORFLOW_HOME}/tensorflow/contrib/lite/python/lite.py
#	e.g.) tflite_convert --output_file=/path/to/saved_model.tflite --saved_model_dir=/path/to/saved_model
#	=> 'saved_model_dir' is a directory of TensorFlow SavedModel.

# TensorFlow TOCO:
#	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md
#	toco:
#		cd ${TENSORFLOW_HOME}
#		bazel build tensorflow/contrib/lite/toco:toco
#		bazel-bin/tensorflow/contrib/lite/toco/toco --help
#	e.g.) toco --input_file=/path/to/optimized_frozen_graph.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=FLOAT --input_type=FLOAT \
#		--input_arrays=input_nodes --output_arrays=output_nodes --input_shapes=1,28,28,1 --output_file=/path/to/saved_model.tflite

import tensorflow as tf
from tensorflow.contrib.lite.python import lite
import os

def tensorflow_lite():
	tf_lite_filepath = './tf_lite_model.tflite'

	# Convert a GraphDef from session.
	#with tf.Session(graph=graph) as sess:
	#	converter = lite.TocoConverter.from_session(sess, in_tensors, out_tensors)
	# Convert a GraphDef from file.
	#converter = lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
	# Convert a SavedModel.
	saved_model_dir_path = './mnist_cnn_saved_model'
	converter = lite.TocoConverter.from_saved_model(saved_model_dir_path)
	# Convert a tf.keras model.
	#converter = lite.TocoConverter.from_keras_model_file(keras_model)

	tf_lite_model = converter.convert()
	with open(tf_lite_filepath, 'wb') as fd:
		fd.write(tf_lite_model)

def main():
	tensorflow_lite()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
