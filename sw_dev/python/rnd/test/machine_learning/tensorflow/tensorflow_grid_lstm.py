# REF [site] >> https://www.tensorflow.org/get_started/get_started

import tensorflow as tf
import numpy as np

#%%-------------------------------------------------------------------

sess = tf.Session()

num_classes = 2
num_hidden_units = 128
tied = False
non_recurrent_fn = None

#cell = tf.contrib.rnn.GridLSTMCell(num_hidden_units, num_frequency_blocks)
cell = tf.contrib.grid_rnn.Grid2LSTMCell(num_hidden_units, tied, non_recurrent_fn)

keep_prob = 0.5

# Unstack: a tensor of shape (samples, time-steps, features) -> a list of 'time-steps' tensors of shape (samples, features).
input_tensor = tf.unstack(input_tensor, num_time_steps, axis=0 if is_time_major else 1)

# Defines a cell.
cell = self._create_unit_cell(num_hidden_units)
cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

# Gets cell outputs.
"""
# REF [site] >> https://www.tensorflow.org/tutorials/recurrent
#cell_state = cell.zero_state(batch_size, tf.float32)
cell_state = tf.zeros([batch_size, cell.state_size])
cell_output_list = []
probabilities = []
loss = 0.0
for inp in input_tensor:
	#cell_output, cell_state = cell(inp, cell_state)
	cell_outputs, _ = cell(inp, cell_state)
	cell_output_list.append(cell_outputs)

	#logits = tf.matmul(cell_output, weights) + biases
	# TODO [check] >>
	logits = tf.layers.dense(cell_output, 1024, activation=tf.nn.softmax, name='fc')
	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
	logits = tf.layers.dropout(logits, rate=dropout_rate, training=is_training, name='dropout')

	probabilities.append(tf.nn.softmax(logits))
	loss += loss_function(probabilities, output_tensor[:, i])
"""
#cell_outputs, cell_state = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)
cell_outputs, _ = tf.nn.static_rnn(cell, input_tensor, dtype=tf.float32)

# Stack: a list of 'time-steps' tensors of shape (samples, features) -> a tensor of shape (samples, time-steps, features).
cell_outputs = tf.stack(cell_outputs, axis=0 if is_time_major else 1)

#with tf.variable_scope('rnn_tf', reuse=tf.AUTO_REUSE):
#	dropout_rate = 1 - keep_prob
#	# NOTE [info] >> If dropout_rate=0.0, dropout layer is not created.
#	cell_outputs = tf.layers.dropout(cell_outputs, rate=dropout_rate, training=is_training, name='dropout')

with tf.variable_scope('fc1', reuse=tf.AUTO_REUSE):
	if 1 == num_classes:
		fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, name='fc')
		#fc1 = tf.layers.dense(cell_outputs, 1, activation=tf.sigmoid, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
	elif num_classes >= 2:
		fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, name='fc')
		#fc1 = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax, activity_regularizer=tf.contrib.layers.l2_regularizer(0.0001), name='fc')
	else:
		assert num_classes > 0, 'Invalid number of classes.'
