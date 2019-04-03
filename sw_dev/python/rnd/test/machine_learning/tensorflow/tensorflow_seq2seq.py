#!/usr/bin/env python

import numpy as np
import tensorflow as tf

def sequence_loss_example():
	#batch_size = 4
	max_time_steps = 5
	num_labels = 3

	num_classes = num_labels + 1  # Adds EOS.
	#num_classes = num_labels

	if 4 == num_classes:
		eos_token = num_classes - 1
		#eos_token = -1  # Error: Received a label value of -1 which is outside the valid range of [0, 4).
		#eos_token = None
	elif 3 == num_classes:
		#eos_token = -1  # Error. Received a label value of -1 which is outside the valid range of [0, 3).
		eos_token = None

	graph = tf.Graph()
	with graph.as_default():
		# NOTE [info] >> logits and targets must have the same first dimension (batch_size * time_steps).

		# (batch_size, max_time_steps, num_classes).
		logits_bm = tf.placeholder(tf.float32, [None, None, num_classes])
		# (batch_size, max_time_steps, num_classes).
		targets = tf.placeholder(tf.int32, [None, None])
		#targets = tf.placeholder(tf.int32, [None, None, num_classes])
		# (batch_size).
		lengths = tf.placeholder(tf.int32, [None])

		#masks = tf.sequence_mask(lengths, maxlen=tf.reduce_max(lengths), dtype=tf.float32)
		masks = tf.sequence_mask(lengths, maxlen=max_time_steps, dtype=tf.float32)

		# Weighted cross-entropy loss for a sequence of logits.
		loss = tf.contrib.seq2seq.sequence_loss(logits=logits_bm, targets=targets, weights=masks)
		#loss = tf.contrib.seq2seq.sequence_loss(logits=logits_bm, targets=tf.argmax(targets, axis=-1), weights=masks)

	def get_probabilities(num_classes, mode=0):
		if 4 == num_classes:
			# When time_steps = 5 and num_classes = 4.
			if 0 == mode:
				# Uses EOS => best.
				return np.array([
					[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
					[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 1.0]]
				])
			elif 1 == mode:
				# Repeat the last probability => not so good.
				return np.array([
					[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
					[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]
				])
			elif 2 == mode:
				# Use zero probability => not so good.
				return np.array([
					[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
					[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
					[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.0, 0.0, 0.0, 0.0]]
				])
		elif 3 == num_classes:
			# When time_steps = 5 and num_classes = 3.
			if 0 == mode:
				return None
			elif 1 == mode:
				# Repeat the last probability => not so good.
				return np.array([
					[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]],
					[[0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
					[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]],
					[[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]
				])
			elif 2 == mode:
				# Use zero probability => not so good.
				return np.array([
					[[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
					[[0.1, 0.2, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
					[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
					[[0.1, 0.2, 0.7], [0.5, 0.2, 0.3], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.0, 0.0, 0.0]]
				])

	def get_dense_targets(max_time_steps, eos_token):
		if 5 == max_time_steps:
			if eos_token is not None:  # When using EOS.
				# When time_steps = 5.
				return np.array([
					[2, 0, eos_token, eos_token, eos_token],
					[2, eos_token, eos_token, eos_token, eos_token],
					[1, 2, eos_token, eos_token, eos_token],
					[2, 0, 1, 2, eos_token]
				])
			else:
				# When time_steps = 5. Not so good.
				return np.array([
					[2, 0, 0, 0, 0],
					[2, 2, 2, 2, 2],
					[1, 2, 2, 2, 2],
					[2, 0, 1, 2, 2]
				])
		elif 4 == max_time_steps:
			if eos_token is not None:  # When using EOS.
				# When time_steps = 4.
				return np.array([
					[2, 0, eos_token, eos_token],
					[2, eos_token, eos_token, eos_token],
					[1, 2, eos_token, eos_token],
					[2, 0, 1, 2]
				])
			else:
				# When time_steps = 4. Not so good.
				return np.array([
					[2, 0, 0, 0],
					[2, 2, 2, 2],
					[1, 2, 2, 2],
					[2, 0, 1, 2]
				])

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		if True:  # When time steps of logits and targets are equal.
			prob_inputs = get_probabilities(num_classes, mode=0)  # max_time_steps = 5.
			dense_targets = get_dense_targets(max_time_steps=max_time_steps, eos_token=eos_token)
		else:  # When time steps of logits and targets are not equal.
			# NOTE [error] >> ValueError: Cannot feed value of shape (4, 5, 3) for Tensor 'Placeholder:0', which has shape '(?, ?, 4)'.
			#	logits and targets must have the same first dimension (batch_size * time_steps).

			prob_inputs = get_probabilities(num_classes, mode=0)  # max_time_steps = 5.
			dense_targets = get_dense_targets(max_time_steps=4, eos_token=eos_token)

		print('Logits =', prob_inputs.shape)
		print(prob_inputs)
		print('Targets =', dense_targets.shape)
		print(dense_targets)

		log_prob_inputs = np.log(prob_inputs)
		sequence_lengths = np.array([2, 1, 2, 4], dtype=np.int32)

		loss1 = sess.run(loss, feed_dict={logits_bm: prob_inputs, targets: dense_targets, lengths: sequence_lengths})
		loss2 = sess.run(loss, feed_dict={logits_bm: log_prob_inputs, targets: dense_targets, lengths: sequence_lengths})
		print('Loss1 = {}, loss2 = {}'.format(loss1, loss2))

# REF [site] >> https://github.com/igormq/ctc_tensorflow_example
# REF [file] >> ${SWL_PYTHON_HOME}/test/machine_learning/tensorflow/run_timit_rnn_ctc.py
def ctc_loss_example():
	batch_size = 4
	max_time_steps = 5
	num_labels = 3
	# The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = num_labels + 1
	blank_label = num_classes - 1
	eos_token = -1
	#eos_token = blank_label

	initial_learning_rate = 1e-2
	momentum = 0.9

	graph = tf.Graph()
	with graph.as_default():
		# [batch_size, max_time_steps, num_classes].
		input_logits_bm = tf.placeholder(tf.float32, [None, None, num_classes])
		# Uses tf.sparse_placeholder() that will generate a SparseTensor required by ctc_loss op.
		targets = tf.sparse_placeholder(tf.int32)
		# 1D array of size [batch_size].
		seq_len = tf.placeholder(tf.int32, [None])

		#shape = tf.shape(input_logits_bm)
		#batch_size, max_time_steps = shape[0], shape[1]
		#input_logits_bm = tf.reshape(input_logits_bm, [batch_size, -1, num_classes])

		# Loss.
		#	Variable-length output.
		input_logits_tm = tf.transpose(input_logits_bm, (1, 0, 2))  # Time-major.
		if True:
			loss = tf.nn.ctc_loss(targets, input_logits_bm, sequence_length=seq_len, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=False)
		else:
			loss = tf.nn.ctc_loss(targets, input_logits_tm, sequence_length=seq_len, preprocess_collapse_repeated=False, ctc_merge_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True)
		cost = tf.reduce_mean(loss)

		#optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)

		# Accuracy.
		#	tf.nn.ctc_beam_search_decoder: it's slower but you'll get better results.
		#	decoded: a list of sparse tensors.
		decoded, log_prob = tf.nn.ctc_beam_search_decoder(input_logits_tm, sequence_length=seq_len, beam_width=100, top_paths=1, merge_repeated=True)  # Time-major.
		# The ctc_greedy_decoder is a special case of the ctc_beam_search_decoder with top_paths=1 and beam_width=1 (but that decoder is faster for this special case).
		#decoded, log_prob = tf.nn.ctc_greedy_decoder(input_logits_tm, sequence_length=seq_len, merge_repeated=True)  # Time-major.

		# Label error rate => inaccuracy.
		#	Variable-length output.
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))  # int64 -> int32.

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		prob_inputs = np.array([
			[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]
		])
		log_prob_inputs = np.log(prob_inputs)
		#dense_target_inputs = np.array([[2, 0, eos_token, eos_token], [2, eos_token, eos_token, eos_token], [1, 2, eos_token, eos_token], [2, 0, 1, 2]])
		#sparse_target_inputs = tf.contrib.layers.dense_to_sparse(dense_target_inputs, eos_token=eos_token)
		sparse_target_inputs = tf.SparseTensorValue(indices=np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3]], dtype=np.int64), values=np.array([2, 0, 2, 1, 2, 2, 0, 1, 2], dtype=np.int64), dense_shape=np.array([4, 4], dtype=np.int64))
		sequence_lengths = np.array([max_time_steps] * batch_size, dtype=np.int32)

		cost, ler, decoded, log_prob = sess.run([cost, ler, tf.sparse.to_dense(decoded[0], default_value=-1), log_prob], feed_dict={input_logits_bm: log_prob_inputs, targets: sparse_target_inputs, seq_len: sequence_lengths})
		print('Cost (ctc_loss) =', cost)
		print('LER (ctc_beam_search_decoder + edit_distance) =', ler)
		print('Decoded (ctc_beam_search_decoder) =\n', decoded)
		print('log(probability) (ctc_beam_search_decoder) =\n', log_prob)
		print('Probability (ctc_beam_search_decoder) =\n', np.exp(log_prob))

		"""
		# For training.
		for epoch in range(num_epochs):
			for step in range(steps_per_epoch):
				batch_cost, _ = sess.run([cost, optimizer], feed_dict=train_feed)
				batch_ler = sess.run(ler, feed_dict=train_feed)
				train_cost += batch_cost * batch_size
				train_ler += batch_ler * batch_size
			val_cost, val_ler = sess.run(input_logits_bm)
		"""

def ctc_loss_v2_example():
	from distutils.version import LooseVersion, StrictVersion
	if LooseVersion(tf.__version__) < LooseVersion('1.13'):
		print('TensorFlow version should be larger than "1.13".')
		return

	batch_size = 4
	max_time_steps = 5
	num_labels = 3
	# The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = num_labels + 1
	blank_label = num_classes - 1
	eos_token = -1
	#eos_token = blank_label

	initial_learning_rate = 1e-2
	momentum = 0.9

	graph = tf.Graph()
	with graph.as_default():
		# [batch_size, max_time_steps, num_classes].
		input_logits_bm = tf.placeholder(tf.float32, [None, None, num_classes])
		# Uses tf.sparse_placeholder() that will generate a SparseTensor required by ctc_loss op.
		targets = tf.sparse_placeholder(tf.int32)
		# 1D array of size [batch_size].
		seq_len = tf.placeholder(tf.int32, [None])

		#shape = tf.shape(input_logits_bm)
		#batch_size, max_time_steps = shape[0], shape[1]
		#input_logits_bm = tf.reshape(input_logits_bm, [batch_size, -1, num_classes])

		# Loss.
		#	Variable-length output.
		input_logits_tm = tf.transpose(input_logits_bm, (1, 0, 2))  # Time-major.
		if True:
			loss = tf.nn.ctc_loss_v2(labels=targets, logits=input_logits_bm, label_length=None, logit_length=seq_len, logits_time_major=False, unique=None, blank_index=blank_label)
		else:
			loss = tf.nn.ctc_loss_v2(labels=targets, logits=input_logits_tm, label_length=None, logit_length=seq_len, logits_time_major=True, unique=None, blank_index=blank_label)
		cost = tf.reduce_mean(loss)

		#optimizer = tf.train.MomentumOptimizer(initial_learning_rate, momentum).minimize(cost)

		# Accuracy.
		#	tf.nn.ctc_beam_search_decoder: it's slower but you'll get better results.
		#	decoded: a list of sparse tensors.
		decoded, log_prob = tf.nn.ctc_beam_search_decoder_v2(input_logits_tm, sequence_length=seq_len, beam_width=100, top_paths=1)  # Time-major.
		# The ctc_greedy_decoder is a special case of the ctc_beam_search_decoder with top_paths=1 and beam_width=1 (but that decoder is faster for this special case).
		#decoded, log_prob = tf.nn.ctc_greedy_decoder(input_logits_tm, sequence_length=seq_len, merge_repeated=True)  # Time-major.

		# Label error rate => inaccuracy.
		#	Variable-length output.
		ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))  # int64 -> int32.

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		prob_inputs = np.array([
			[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
			[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]
		])
		log_prob_inputs = np.log(prob_inputs)
		#dense_target_inputs = np.array([[2, 0, eos_token, eos_token], [2, eos_token, eos_token, eos_token], [1, 2, eos_token, eos_token], [2, 0, 1, 2]])
		#sparse_target_inputs = tf.contrib.layers.dense_to_sparse(dense_target_inputs, eos_token=eos_token)
		sparse_target_inputs = tf.SparseTensorValue(indices=np.array([[0, 0], [0, 1], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1], [3, 2], [3, 3]], dtype=np.int64), values=np.array([2, 0, 2, 1, 2, 2, 0, 1, 2], dtype=np.int64), dense_shape=np.array([4, 4], dtype=np.int64))
		sequence_lengths = np.array([max_time_steps] * batch_size, dtype=np.int32)

		cost, ler, decoded, log_prob = sess.run([cost, ler, tf.sparse.to_dense(decoded[0], default_value=-1), log_prob], feed_dict={input_logits_bm: log_prob_inputs, targets: sparse_target_inputs, seq_len: sequence_lengths})
		print('Cost (ctc_loss_v2) =', cost)
		print('LER (ctc_beam_search_decoder_v2 + edit_distance) =', ler)
		print('Decoded (ctc_beam_search_decoder_v2) =\n', decoded)
		print('log(probability) (ctc_beam_search_decoder_v2) =\n', log_prob)
		print('Probability (ctc_beam_search_decoder_v2) =\n', np.exp(log_prob))

		"""
		# For training.
		for epoch in range(num_epochs):
			for step in range(steps_per_epoch):
				batch_cost, _ = sess.run([cost, optimizer], feed_dict=train_feed)
				batch_ler = sess.run(ler, feed_dict=train_feed)
				train_cost += batch_cost * batch_size
				train_ler += batch_ler * batch_size
			val_cost, val_ler = sess.run(input_logits_bm)
		"""

# REF [site] >> https://stackoverflow.com/questions/45482813/tensorflow-cant-understand-ctc-beam-search-decoder-output-sequence
def ctc_beam_search_decoder_example_1():
	batch_size = 4
	max_time_steps = 5
	num_labels = 3
	# The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = num_labels + 1
	blank_label = num_classes - 1
	#eos_token = -1
	#eos_token = blank_label

	graph = tf.Graph()
	with graph.as_default():
		input_probs_bm = tf.placeholder(tf.float32, shape=(batch_size, max_time_steps, num_classes))
		input_probs_tm = tf.transpose(input_probs_bm, perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes].
		logits_tm = tf.log(input_probs_tm)

		sequence_lengths = [max_time_steps] * batch_size
		decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(logits_tm, sequence_length=sequence_lengths, beam_width=3, top_paths=1, merge_repeated=True)  # Time-major.

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
			prob_inputs = np.array([
				[[0.1, 0.1, 0.8, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]],
				[[0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
				[[0.1, 0.7, 0.2, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]],
				[[0.1, 0.2, 0.7, 0.0], [0.5, 0.2, 0.3, 0.0], [0.3, 0.6, 0.1, 0.0], [0.1, 0.2, 0.7, 0.0], [0.1, 0.2, 0.7, 0.0]]
			])

		decoded, log_probabilities = sess.run([decoded, log_probabilities], feed_dict={input_probs_bm: prob_inputs})
		#print('decoded =\n', decoded, sep='')
		print('log_probabilities =\n', log_probabilities, sep='')

		for idx, stv in enumerate(decoded):
			#print('Decoded path {} =\n{}'.format(idx, tf.sparse.to_dense(stv, default_value=-1).eval(session=sess)))
			print('Decoded path {} =\n{}'.format(idx, tf.sparse_to_dense(sparse_indices=stv.indices, output_shape=stv.dense_shape, sparse_values=stv.values, default_value=-1).eval(session=sess)))

# REF [site] >> https://programtalk.com/python-examples/tensorflow.nn.ctc_beam_search_decoder/
def ctc_beam_search_decoder_example_2():
	batch_size = 1
	num_labels = 5
	# The largest value (num_classes - 1) is reserved for the blank label.
	num_classes = num_labels + 1
	blank_label = num_classes - 1
	#eos_token = -1
	#eos_token = blank_label

	input_prob_matrix = np.asarray(
		[[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
		[0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
		[0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
		[0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
		[0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878]],
		dtype=np.float32)
	# Add arbitrary offset.
	input_log_prob_matrix = np.log(input_prob_matrix) + 2.0

	# An array of shape (batch_size, max_time_steps, num_classes).
	inputs = ([row[np.newaxis, :] for row in input_log_prob_matrix] + 2 * [np.zeros((1, num_classes), dtype=np.float32)])  # Pad to zeros.
	inputs = np.reshape(inputs, (batch_size, -1, num_classes))
	max_time_steps = inputs.shape[1]

	# A vector of length batch_size which has sequence_length.
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
		input_logits_bm = tf.placeholder(tf.float32, shape=(batch_size, max_time_steps, num_classes))
		input_logits_tm = tf.transpose(input_logits_bm, perm=[1, 0, 2])  # TF expects dimensions [max_time, batch_size, num_classes].
		decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(input_logits_tm, sequence_length=seq_lens, beam_width=2, top_paths=2, merge_repeated=True)

	with tf.Session(graph=graph) as sess:
		tf.global_variables_initializer().run()

		decoded, log_probabilities = sess.run([decoded, log_probabilities], feed_dict={input_logits_bm: inputs})
		#print('decoded =\n', decoded)
		print('log_probabilities =', log_probabilities)

		for idx, stv in enumerate(decoded):
			#print('Decoded path {} = {}'.format(idx, tf.sparse.to_dense(stv, default_value=-1).eval(session=sess)))
			print('Decoded path {} = {}'.format(idx, tf.sparse_to_dense(sparse_indices=stv.indices, output_shape=stv.dense_shape, sparse_values=stv.values, default_value=-1).eval(session=sess)))

def main():
	#sequence_loss_example()

	# CTC loss.
	ctc_loss_example()
	ctc_loss_v2_example()

	#ctc_beam_search_decoder_example_1()
	#ctc_beam_search_decoder_example_2()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
