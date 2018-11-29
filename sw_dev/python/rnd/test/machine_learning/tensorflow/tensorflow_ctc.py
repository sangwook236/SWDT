#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example
# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/tensorflow/run_ctc_example.py
def ctc_loss_example():
	num_classes = 4  # num_classes = num_labels + 1. The largest value (num_classes - 1) is reserved for the blank label.
	initial_learning_rate = 1e-2
	momentum = 0.9

	graph = tf.Graph()
	with graph.as_default():
		# [batch_size, max_time_steps, num_classes].
		input_logits = tf.placeholder(tf.float32, [None, None, num_classes])
		# Uses tf.sparse_placeholder() that will generate a SparseTensor required by ctc_loss op.
		targets = tf.sparse_placeholder(tf.int32)
		# 1D array of size [batch_size].
		seq_len = tf.placeholder(tf.int32, [None])

		#shape = tf.shape(input_logits)
		#batch_size, max_time_steps = shape[0], shape[1]
		#input_logits = tf.reshape(input_logits, [batch_size, -1, num_classes])

		input_logits0 = tf.transpose(input_logits, (1, 0, 2))  # Time-major.

		# Loss.
		#	Variable-length output.
		loss = tf.nn.ctc_loss(targets, input_logits0, sequence_length=seq_len, ctc_merge_repeated=True, time_major=True)
		cost = tf.reduce_mean(loss)

		#optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)

		# Accuracy.
		#	tf.nn.ctc_beam_search_decoder: it's slower but you'll get better results.
		#	decoded: a list of sparse tensors.
		#decoded, log_prob = tf.nn.ctc_beam_search_decoder(input_logits0, sequence_length=seq_len, beam_width=100, top_paths=1, merge_repeated=True)  # Time-major.
		decoded, log_prob = tf.nn.ctc_greedy_decoder(input_logits0, sequence_length=seq_len, merge_repeated=True)  # Time-major.

		# Label error rate => inaccuracy.
		#	Variable-length output.
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		batch_size = 4
		max_time_steps = 5
		prob_inputs = np.array([[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]])
		log_prob_inputs = np.log(prob_inputs)
		#dense_target_inputs = np.array([[2, 0, -1, -1], [2, -1, -1, -1], [1, 2, -1, -1], [2, 0, 1, 2]])
		#sparse_target_inputs = tf.contrib.layers.dense_to_sparse(dense_target_inputs, eos_token=-1)
		sparse_target_inputs = tf.SparseTensorValue(indices=np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3]], dtype=np.int64), values=np.array([2, 0, 2, 1, 2, 2, 0, 1, 2], dtype=np.int64), dense_shape=np.array([4, 4], dtype=np.int64))
		sequence_lengths = np.array([max_time_steps] * batch_size, dtype=np.int32)

		cost, ler = sess.run([cost, ler], feed_dict={input_logits: log_prob_inputs, targets: sparse_target_inputs, seq_len: sequence_lengths})
		print('cost =', cost)
		print('ler =', ler)

		"""
		# For training.
		for epoch in range(num_epochs):
			for step in range(steps_per_epoch):
				batch_cost, _ = sess.run([cost, optimizer], feed_dict=train_feed)
				batch_ler = sess.run(ler, feed_dict=train_feed)
				train_cost += batch_cost * batch_size
				train_ler += batch_ler * batch_size
			val_cost, val_ler = sess.run([cost, ler], feed_dict=val_feed)
		"""

# REF [site] >> https://stackoverflow.com/questions/45482813/tensorflow-cant-understand-ctc-beam-search-decoder-output-sequence
def ctc_beam_search_decoder_example_1():
	batch_size = 4
	max_time_steps = 5
	num_classes = 4  # num_classes = num_labels + 1. The largest value (num_classes - 1) is reserved for the blank label.

	graph = tf.Graph()
	with graph.as_default():
		input_probs = tf.placeholder(tf.float32, shape=(batch_size, max_time_steps, num_classes))
		input_probs_transposed = tf.transpose(input_probs, perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes].
		logits = tf.log(input_probs_transposed)

		sequence_lengths = [max_time_steps] * batch_size
		decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(logits, sequence_length=sequence_lengths, beam_width=3, top_paths=1, merge_repeated=False)  # Time-major.

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		if False:
			np.random.seed(7)
			r = np.random.randint(0, 100, size=(batch_size, max_time_steps, num_classes))
			prob_inputs = r / np.sum(r, 2).reshape(batch_size, max_time_steps, 1)
		elif False:
			np.random.seed(50)
			r = np.random.randint(0, 100, size=(batch_size, max_time_steps, num_classes))
			prob_inputs = r / np.sum(r, 2).reshape(batch_size, max_time_steps, 1)
		else:
			prob_inputs = np.array([[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
				[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
				[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
				[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]])

		decoded, log_probabilities = sess.run([decoded, log_probabilities], feed_dict={input_probs: prob_inputs})
		#print('decoded =\n', decoded, sep='')
		print('log_probabilities =\n', log_probabilities, sep='')

		for idx, stv in enumerate(decoded):
			#print('Decoded path {} =\n{}'.format(idx, tf.sparse.to_dense(stv, default_value=-1).eval(session=sess)))
			print('Decoded path {} =\n{}'.format(idx, tf.sparse_to_dense(sparse_indices=stv.indices, output_shape=stv.dense_shape, sparse_values=stv.values, default_value=-1).eval(session=sess)))

# REF [site] >> https://programtalk.com/python-examples/tensorflow.nn.ctc_beam_search_decoder/
def ctc_beam_search_decoder_example_2():
	batch_size = 1
	num_classes = 6  # num_classes = num_labels + 1. The largest value (num_classes - 1) is reserved for the blank label.

	input_prob_matrix = np.asarray(
		[[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
		[0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
		[0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
		[0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
		[0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878]],
		dtype=np.float32)
	# Add arbitrary offset.
	input_log_prob_matrix = np.log(input_prob_matrix) + 2.0

	# max_time_steps array of batch_size x num_classes matrices
	inputs = ([row[np.newaxis, :] for row in input_log_prob_matrix] + 2 * [np.zeros((1, num_classes), dtype=np.float32)])  # Pad to zeros.
	inputs = np.reshape(inputs, (batch_size, -1, num_classes))
	max_time_steps = inputs.shape[1]

	# batch_size length vector of sequence_lengths
	seq_lens = np.array([max_time_steps] * batch_size, dtype=np.int32)

	"""
	# batch_size length vector of negative log probabilities.
	log_prob_truth = np.array([
		0.584855,  # Output beam 0.
		0.389139  # Output beam 1.
	], np.float32)[np.newaxis, :]

	# decode_truth: two SparseTensors, (indices, values, shape).
	decode_truth = [
		# Beam 0, batch 0, two outputs decoded.
		(np.array([[0, 0], [0, 1]], dtype=np.int64),
		np.array([1, 0], dtype=np.int64),
		np.array([1, 2], dtype=np.int64)),
		# Beam 1, batch 0, three outputs decoded.
		(np.array([[0, 0], [0, 1], [0, 2]], dtype=np.int64),
		np.array([0, 1, 0], dtype=np.int64),
		np.array([1, 3], dtype=np.int64)),
	]
	"""

	graph = tf.Graph()
	with graph.as_default():
		input_logits = tf.placeholder(tf.float32, shape=(batch_size, max_time_steps, num_classes))
		input_logits0 = tf.transpose(input_logits, perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes].
		decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(input_logits0, sequence_length=seq_lens, beam_width=2, top_paths=2, merge_repeated=True)

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		decoded, log_probabilities = sess.run([decoded, log_probabilities], feed_dict={input_logits: inputs})
		#print('decoded =\n', decoded)
		print('log_probabilities =', log_probabilities)

		for idx, stv in enumerate(decoded):
			#print('Decoded path {} = {}'.format(idx, tf.sparse.to_dense(stv, default_value=-1).eval(session=sess)))
			print('Decoded path {} = {}'.format(idx, tf.sparse_to_dense(sparse_indices=stv.indices, output_shape=stv.dense_shape, sparse_values=stv.values, default_value=-1).eval(session=sess)))

def main():
	ctc_loss_example()

	ctc_beam_search_decoder_example_1()
	ctc_beam_search_decoder_example_2()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
