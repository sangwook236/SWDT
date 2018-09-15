#!/usr/bin/env python

# REF [site] >>
#	https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

# Checkpoint:
#	https://www.tensorflow.org/guide/checkpoints
# Structure of a checkpoint directory:
#	checkpoint
#		Checkpoint info.
#	tf_ckpt-????.data-?????-of-?????
#		Weights of variables.
#	tf_ckpt-????.index
#	tf_ckpt-????.meta
#		Meta graph.

# SavedModel:
#	https://www.tensorflow.org/guide/saved_model
#	https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model
# Structure of a SavedModel directory:
#	https://www.tensorflow.org/guide/saved_model#structure_of_a_savedmodel_directory
#	assets/
#	assets.extra/
#	variables/
#		variables.data-?????-of-?????
#		variables.index
#	saved_model.pb or saved_model.pbtxt
# SavedModel Command Line Interface (CLI):
#	https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel
#	saved_model_cli show --dir /path/to/saved_model
#	saved_model_cli show --dir /path/to/saved_model --tag_set serve
#	saved_model_cli show --dir /path/to/saved_model --tag_set serve,gpu
#	saved_model_cli show --dir /path/to/saved_model --tag_set serve --signature_def serving_default
#	saved_model_cli show --dir /path/to/saved_model --all

# TensorFlow Serving:
#	https://www.tensorflow.org/serving/
#	https://www.tensorflow.org/serving/serving_basic
#	https://www.tensorflow.org/serving/signature_defs
#	https://github.com/tensorflow/serving

import tensorflow as tf
import os

def save_model_to_checkpoint():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_checkpoint_filename_prefix = 'tf_ckpt'

	graph = tf.Graph()
	with graph.as_default():
		w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
		w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')

	saver = tf.train.Saver()
	#saver = tf.train.Saver([w1, w2])
	# Save a model every 2 hours and maximum 4 latest models are saved.
	#saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		sess.run(tf.global_variables_initializer())

		# Do something with the model.

		model_saved_path = saver.save(sess, os.path.join(model_checkpoint_dir_path, model_checkpoint_filename_prefix))
		#model_saved_path = saver.save(sess, 'my-model', global_step=step, meta_graph_suffix='meta', write_meta_graph=True, write_state=True)

def load_model_from_checkpoint():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'tf_ckpt-7740.meta'

	graph = tf.Graph()
	with graph.as_default():
		# Method 1.
		#	Create a model.
		#saver = tf.train.Saver()

		# Method 2.
		#	Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(model_checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_dir_path))

		# Save the graph and model to a SavedModel.
		#	REF [function] >> checkpoint_to_saved_model()

def checkpoint_to_graph():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'tf_ckpt-7740.meta'
	graph_dir_path = '.'
	graph_bin_filename = 'mnist_cnn_graph.pb'
	graph_txt_filename = 'mnist_cnn_graph.pbtxt'
	# For Mobis-postech project.
	#model_checkpoint_dir_path = './mobis_postech_checkpoint'
	#model_meta_graph_filename = 'weight_99_4cls.meta'
	#graph_dir_path = '.'
	#graph_bin_filename = 'mobis_postech_graph.pb'
	#graph_txt_filename = 'mobis_postech_graph.pbtxt'

	graph = tf.Graph()
	with graph.as_default():
		# Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		#ckpt = tf.train.get_checkpoint_state(model_checkpoint_dir_path)
		#saver.restore(sess, ckpt.model_checkpoint_path)
		##saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_dir_path))

		# Save a graph.
		tf.train.write_graph(sess.graph_def, graph_dir_path, graph_bin_filename, as_text=False)
		tf.train.write_graph(sess.graph_def, graph_dir_path, graph_txt_filename, as_text=True)

# REF [site] >> https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
def checkpoint_to_saved_model():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'tf_ckpt-7740.meta'
	saved_model_dir_path = './mnist_cnn_saved_model'

	graph = tf.Graph()
	with graph.as_default():
		# Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(model_checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_dir_path))

		#operations = graph.get_operations()
		##print(operations)
		#for op in operations:
		#	#print(op)
		#	print(op.name)
		input_tensor = graph.get_tensor_by_name('input_tensor_ph:0')
		output_tensor = graph.get_tensor_by_name('mnist_cnn_using_tf/fc2/fc/Softmax:0')

		# Build the SignatureDef map.
		#	REF [site] >> https://www.tensorflow.org/serving/signature_defs
		prediction_inputs = tf.saved_model.utils.build_tensor_info(input_tensor)
		prediction_outputs = tf.saved_model.utils.build_tensor_info(output_tensor)
		prediction_signature = (
			tf.saved_model.signature_def_utils.build_signature_def(
				inputs={'images': prediction_inputs},
				outputs={'scores': prediction_outputs},
				method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
			)
		)

		# Save as a SavedModel.
		builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir_path)
		builder.add_meta_graph_and_variables(
			sess, [tf.saved_model.tag_constants.SERVING], saver=saver,
			signature_def_map={
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
			},
			main_op=tf.tables_initializer(),
			strip_default_attrs=True
		)
		builder.save(as_text=False)

def graph_to_tensorboard_log():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'tf_ckpt-7740.meta'
	log_dir_path = './tensorboard_log'
	# For Mobis-postech project.
	#model_checkpoint_dir_path = './mobis_postech_checkpoint'
	#model_meta_graph_filename = 'weight_99_4cls.meta'
	#log_dir_path = './tensorboard_mobis_postech_log'

	graph = tf.Graph()
	with graph.as_default():
		# Method 1.
		#	Create a model.

		# Method 2.
		#	Load a graph.
		tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		operations = graph.get_operations()
		#print(operations)
		for op in operations:
			#print(op)
			print(op.name)

		# Write log.
		file_writer = tf.summary.FileWriter(logdir=log_dir_path, graph=graph) 
		#file_writer.flush()
		#file_writer.close()

# Frozen graph:
#	REF [site] >>
#		https://www.tensorflow.org/extend/tool_developers/#freezing
#		https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
#	NOTE [info] >>
#		Use TensorBoard or the summarize_graph tool to get a graph's shape.
#		If there are still unsupported layers, check out graph_transform tools.
#	${TENSORFLOW_HOME}/tensorflow/python/tools/freeze_graph.py
#		The freeze_graph tool takes a graph definition and a set of checkpoints and freezes them together into a single file.
#		freeze_graph
#			cd ${TENSORFLOW_HOME}
#			bazel build tensorflow/python/tools:freeze_graph
#			bazel-bin/tensorflow/python/tools/freeze_graph --help
#		python freeze_graph.py
#			${TENSORFLOW_HOME}/tensorflow/python/tools/freeze_graph.py
#		e.g.) freeze_graph --input_graph=/path/to/graph.pbtxt --input_binary=false --input_checkpoint=/path/to/checkpoint/tf_ckpt-1234 --output_graph=/path/to/frozen_graph.pb --output_node_names=output_nodes
#			${TENSORFLOW_HOME}/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=/home/sangwook/work/mnist_cnn_graph.pbtxt --input_binary=false --input_checkpoint=/home/sangwook/work/mnist_cnn_checkpoint/tf_ckpt-7740 --output_graph=/home/sangwook/work/mnist_cnn_frozen_graph.pb --output_node_names=mnist_cnn_using_tf/fc2/fc/Softmax
#		=> Recommend using TensorBoard to get 'output_node_names' which I think is operations' names in a TensorFlow graph.
#	${TENSORFLOW_HOME}/tensorflow/python/tools/optimize_for_inference.py
#		The optimize_for_inference tool takes in the input and output names and does another pass to strip out unnecessary layers.
#		optimize_for_inference
#			cd ${TENSORFLOW_HOME}
#			bazel build tensorflow/python/tools:optimize_for_inference
#			bazel-bin/tensorflow/python/tools/optimize_for_inference --help
#		python optimize_for_inference.py
#			${TENSORFLOW_HOME}/tensorflow/python/tools/optimize_for_inference.py
#		e.g.) optimize_for_inference --input=/path/to/frozen_graph.pb --output=/path/to/optimized_frozen_graph.pb --frozen_graph=true --input_names=input_nodes --output_names=output_nodes
#	REF [site] >>
#		https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3
def freeze_graph():
	# Use the freeze_graph tool.
	raise NotImplementedError

from tensorflow.contrib.lite.python import lite

# TensorFlow Lite:
#	REF [site] >>
#		https://www.tensorflow.org/mobile/tflite/
#		https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite
#		https://www.tensorflow.org/mobile/tflite/demo_android
#		https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/README.md
# Convert a model to tflite:
#	${TENSORFLOW_HOME}/tensorflow/contrib/lite/python/lite.py
#	${TENSORFLOW_HOME}/tensorflow/contrib/lite/python/tflite_convert.py
#		tflite_convert
#			cd ${TENSORFLOW_HOME}
#			bazel build tensorflow/contrib/lite/python:tflite_convert
#			bazel-bin/tensorflow/contrib/lite/python/tflite_convert --help
#		python tflite_convert.py
#			${TENSORFLOW_HOME}/tensorflow/contrib/lite/python/tflite_convert.py
#		e.g.) tflite_convert --output_file=/path/to/saved_model.tflite --saved_model_dir=/path/to/saved_model
#		=> 'saved_model_dir' is a directory of TensorFlow SavedModel.
#	${TENSORFLOW_HOME}/tensorflow/contrib/lite/toco/
#		toco
#			cd ${TENSORFLOW_HOME}
#			bazel build tensorflow/contrib/lite/toco:toco
#			bazel-bin/tensorflow/contrib/lite/toco/toco --help
#		e.g.) toco --input_file=/path/to/optimized_frozen_graph.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=FLOAT --input_type=FLOAT \
#			--input_arrays=input_nodes --output_arrays=output_nodes --input_shapes=1,28,28,1 --output_file=/path/to/saved_model.tflite
#	REF [site] >>
#		https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md
#		https://medium.com/tensorflow/using-tensorflow-lite-on-android-9bbc9cb7d69d
#		https://heartbeat.fritz.ai/intro-to-machine-learning-on-android-how-to-convert-a-custom-model-to-tensorflow-lite-e07d2d9d50e3
def tensorflow_lite():
	# Converting a GraphDef from session.
	#converter = lite.TocoConverter.from_session(sess, in_tensors, out_tensors)
	# Converting a GraphDef from file.
	#converter = lite.TocoConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
	# Converting a SavedModel.
	saved_model_dir = './mnist_cnn_checkpoint'
	converter = lite.TocoConverter.from_saved_model(saved_model_dir)
	# Converting a tf.keras model.
	#converter = lite.TocoConverter.from_keras_model_file(keras_model)

	tflite_model = converter.convert()
	with open('converted_model.tflite', 'wb') as fd:
		fd.write(tflite_model)

def main():
	#save_model_to_checkpoint()  # Actually this is not run.
	#load_model_from_checkpoint()

	checkpoint_to_graph()
	#freeze_graph()  # Not yet implemented.

	graph_to_tensorboard_log()

	# TensorFlow SavedModel.
	#checkpoint_to_saved_model()

	# TensorFlow Lite.
	#tensorflow_lite()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
