#!/usr/bin/env python

# REF [site] >>
#	https://www.tensorflow.org/guide/saved_model
#	https://www.tensorflow.org/guide/checkpoints
#	https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

# In a checkpoint directory:
#	checkpoint
#	model.ckpt-1042.data-00000-of-00001
#	model.ckpt-1042.index
#	model.ckpt-1042.meta

# REF [site] >>
#	https://www.tensorflow.org/serving/
#	https://www.tensorflow.org/serving/serving_basic

# SavedModel Command Line Interface (CLI):
#	saved_model_cli show --dir /path/to/serving_model
#	saved_model_cli show --dir /path/to/serving_model --tag_set serve
#	saved_model_cli show --dir /path/to/serving_model --tag_set serve,gpu
#	saved_model_cli show --dir /path/to/serving_model --tag_set serve --signature_def serving_default
#	saved_model_cli show --dir /path/to/serving_model --all
#	REF [site] >> https://www.tensorflow.org/guide/saved_model
#		"CLI to inspect and execute SavedModel":

import tensorflow as tf
import os

def save_model_to_checkpoint():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_checkpoint_filename_prefix = 'model.ckpt'

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
		sess.run(tf.global_variables_initializer())

		# Do something with the model.

		model_saved_path = saver.save(sess, os.path.join(model_checkpoint_dir_path, model_checkpoint_filename_prefix))
		#model_saved_path = saver.save(sess, 'my-model', global_step=step, meta_graph_suffix='meta', write_meta_graph=True, write_state=True)

def load_model_from_checkpoint():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'model.meta'
	graph_dir_path = '.'

	graph = tf.Graph()
	with graph.as_default():
		# Method 1.
		# Create a model.
		#saver = tf.train.Saver()

		# Method 2.
		saver = tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(model_checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_dir_path))

		# Save graph and serving model.
		#	REF [function] >> checkpoint_to_serving_model()

def checkpoint_to_serving_model():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'model.meta'
	serving_model_dir_path = './mnist_cnn_serving_model'
	graph_dir_path = '.'

	graph = tf.Graph()
	with graph.as_default():
		saver = tf.train.import_meta_graph(os.path.join(model_checkpoint_dir_path, model_meta_graph_filename))

	#config = tf.ConfigProto()
	#with tf.Session(graph=graph, config=config) as sess:
	with tf.Session(graph=graph) as sess:
		# Load a model checkpoint.
		ckpt = tf.train.get_checkpoint_state(model_checkpoint_dir_path)
		saver.restore(sess, ckpt.model_checkpoint_path)
		#saver.restore(sess, tf.train.latest_checkpoint(model_checkpoint_dir_path))

		# Save a graph.
		tf.train.write_graph(sess.graph_def, graph_dir_path, 'mnist_cnn_graph.pb', as_text=False)
		tf.train.write_graph(sess.graph_def, graph_dir_path, 'mnist_cnn_graph.pbtxt', as_text=True)

		# Save a serving model.
		builder = tf.saved_model.builder.SavedModelBuilder(serving_model_dir_path)
		builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], saver=saver)
		builder.save(as_text=False)

def main():
	save_model_to_checkpoint()
	load_model_from_checkpoint()

	# TensorFlow Serving.
	checkpoint_to_serving_model()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
