#!/usr/bin/env python

# REF [site] >>
#	https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

# Description of saved models:
#	REF [site] >> https://www.tensorflow.org/mobile/prepare_models

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
	checkpoint_dir_path = './mnist_cnn_checkpoint'
	checkpoint_filename_prefix = 'tf_ckpt'

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

		model_saved_path = saver.save(sess, os.path.join(checkpoint_dir_path, checkpoint_filename_prefix))
		#model_saved_path = saver.save(sess, 'my-model', global_step=step, meta_graph_suffix='meta', write_meta_graph=True, write_state=True)

def load_model_from_checkpoint():
	checkpoint_dir_path = './mnist_cnn_checkpoint'
	checkpoint_meta_graph_filename = 'tf_ckpt-7740.meta'

	graph = tf.Graph()
	with graph.as_default():
		# Method 1.
		#	Create a model.
		#saver = tf.train.Saver()

		# Method 2.
		#	Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir_path, checkpoint_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))

		# Save the graph and model to a SavedModel.
		#	REF [function] >> checkpoint_to_saved_model()

	if True:
		with tf.Session(graph=graph) as sess:
			input_tensor = sess.graph.get_tensor_by_name('input_tensor_ph:0')
			output_tensor = sess.graph.get_tensor_by_name('mnist_cnn_using_tf/fc2/dense/Softmax:0')
			print('input_tensor =', input_tensor.get_shape())
			print('output_tensor =', output_tensor.get_shape())

# Inspect checkpoint:
#	python inspect_checkpoint.py:
#		${TENSORFLOW_HOME}/tensorflow/python/tools/inspect_checkpoint.py
#	e.g.) python inspect_checkpoint.py --file_name=checkpoint_filename
#		python inspect_checkpoint.py --file_name=/home/sangwook/work/mnist_cnn_checkpoint/tf_ckpt-7740
#		python inspect_checkpoint.py --file_name=/home/sangwook/work/mnist_cnn_checkpoint/tf_ckpt-7740 --all_tensor_names=True
#		python inspect_checkpoint.py --file_name=/home/sangwook/work/mnist_cnn_checkpoint/tf_ckpt-7740 --all_tensors=True
def inspect_checkpoint_tool():
	raise NotImplementedError('Use the inspect_checkpoint tool')

def checkpoint_to_graph():
	checkpoint_dir_path = './mnist_cnn_checkpoint'
	checkpoint_meta_graph_filename = 'tf_ckpt-7740.meta'
	graph_dir_path = '.'
	graph_bin_filename = 'mnist_cnn_graph.pb'
	graph_txt_filename = 'mnist_cnn_graph.pbtxt'

	graph = tf.Graph()
	with graph.as_default():
		# Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir_path, checkpoint_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		#ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		#saver.restore(sess, ckpt.model_checkpoint_path)
		##saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))

		# Save a graph.
		tf.train.write_graph(sess.graph_def, graph_dir_path, graph_bin_filename, as_text=False)
		tf.train.write_graph(sess.graph_def, graph_dir_path, graph_txt_filename, as_text=True)

# REF [site] >> https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py
def checkpoint_to_saved_model():
	checkpoint_dir_path = './mnist_cnn_checkpoint'
	checkpoint_meta_graph_filename = 'tf_ckpt-7740.meta'
	saved_model_dir_path = './mnist_cnn_saved_model'

	graph = tf.Graph()
	with graph.as_default():
		# Load a graph.
		saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir_path, checkpoint_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir_path))

		#operations = graph.get_operations()
		##print(operations)
		#for op in operations:
		#	#print(op)
		#	print(op.name)
		input_tensor = graph.get_tensor_by_name('input_tensor_ph:0')
		output_tensor = graph.get_tensor_by_name('mnist_cnn_using_tf/fc2/dense/Softmax:0')

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

def main():
	#save_model_to_checkpoint()  # Actually this is not run.
	#load_model_from_checkpoint()

	#inspect_checkpoint_tool()  # Not implemented.
	checkpoint_to_graph()

	# TensorFlow SavedModel.
	#checkpoint_to_saved_model()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
