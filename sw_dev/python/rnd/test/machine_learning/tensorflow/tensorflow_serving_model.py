#!/usr/bin/env python

# REF [site] >>
#	https://www.tensorflow.org/serving/serving_basic
#	https://www.tensorflow.org/guide/saved_model
#		"CLI to inspect and execute SavedModel":
#			saved_model_cli show --dir /path/to/serving_model
#			saved_model_cli show --dir /path/to/serving_model --tag_set serve
#			saved_model_cli show --dir /path/to/serving_model --tag_set serve,gpu
#			saved_model_cli show --dir /path/to/serving_model --tag_set serve --signature_def serving_default
#			saved_model_cli show --dir /path/to/serving_model --all

import tensorflow as tf
import os

def checkpoint2servingmodel():
	model_checkpoint_dir_path = './mnist_cnn_checkpoint'
	model_meta_graph_filename = 'model.meta'
	serving_model_dir_path = './mnist_cnn_serving_model'
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

		# Save a graph.
		tf.train.write_graph(sess.graph_def, graph_dir_path, 'mnist_cnn_graph.pb', as_text=False)
		tf.train.write_graph(sess.graph_def, graph_dir_path, 'mnist_cnn_graph.pbtxt', as_text=True)

		# Save a serving model.
		builder = tf.saved_model.builder.SavedModelBuilder(serving_model_dir_path)
		builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], saver=saver)
		builder.save(as_text=False)

def main():
	checkpoint2servingmodel()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
