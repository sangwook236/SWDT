import tensorflow as tf
from PIL import Image
import numpy as np
import math
import time

#%%------------------------------------------------------------------
# An example of 2D GridRNN.

num_dims = 2
num_classes = 2
num_examples = 10
batch_size = 4
num_epochs = 500
if True:
	num_hidden_units = 256
	num_layers = None
else:
	num_hidden_units = 32
	num_layers = 4  # Error!!!
keep_prob = 0.8

input_shape = (None, 20, 30, 3)  # (samples, height, width, features(channels)).
ouput_shape = input_shape[:3] + (num_classes,)

x_ph = tf.placeholder(tf.float32, input_shape)
y_ph = tf.placeholder(tf.float32, ouput_shape)

def to_one_hot_encoding(label_indexes, num_classes):
	return np.eye(num_classes)[label_indexes].reshape(label_indexes.shape + (-1,))

# Generates random data.
x = np.random.rand(num_examples, input_shape[1], input_shape[2], input_shape[3])
y_gt = np.random.rand(num_examples, ouput_shape[1], ouput_shape[2], ouput_shape[3])
y_gt = to_one_hot_encoding(np.argmax(y_gt, axis=-1), num_classes)

def create_unit_cell(num_hidden_units, num_dims, tied=False, non_recurrent_fn=None):
	#num_frequency_blocks = [1]
	#start_freqindex_list = [0]
	#end_freqindex_list = [1]
	#return tf.contrib.rnn.GridLSTMCell(num_hidden_units, num_frequency_blocks=num_frequency_blocks, start_freqindex_list=start_freqindex_list, end_freqindex_list=end_freqindex_list, reuse=True)

	# NOTICE [info] >> tf.contrib.grid_rnn.Grid2LSTMCell receives input data from 0-th dimension only.
	#return tf.contrib.grid_rnn.Grid2LSTMCell(num_hidden_units, tied, non_recurrent_fn)

	#def cell_fn(num_hidden_units):
	#	return tf.contrib.rnn.LSTMCell(num_units=num_hidden_units, forget_bias=1.0, use_peepholes=False)
	# NOTICE [info] >> 'input_dims=None & output_dims=None' means that there are no input and no output.
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=None, output_dims=None, priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=cell_fn, non_recurrent_fn=non_recurrent_fn)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=[0, 1], output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=cell_fn, non_recurrent_fn=non_recurrent_fn)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=[0, 1], output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=0, output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)
	return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=0, output_dims=0, priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)

def build_grid_cell_model(x, num_dims, num_hidden_units, num_layers, keep_prob):
	#import pdb
	#pdb.set_trace()

	input_dims = [0] #[0, 1]
	output_dims = [0] #[0, 1]
	num_input_dims = len(input_dims)
	num_output_dims = len(output_dims)

	cell = create_unit_cell(num_hidden_units, num_dims)
	if num_layers is not None and num_layers > 1:
		cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

	# Input of 1 dimension = an 1-vector of the same length of the feature vector.
	# Input of 2 dimensions = an 1-vector of twice the length of the feature vector.
	# Input of n dimensions = an 1-vector of an n-fold length of the feature vector. (num_input_dims * features).

	# A tensor of shape (samples, height, width, features) -> a list of 'height * width' tensors of shape (samples, features).
	x_shape = x.shape.as_list()
	# (samples, height, width, features) -(?)-> (samples, height * width, num_input_dims, features) or (samples, height * width, num_input_dims * features).
	x = tf.reshape(x, (-1, x_shape[1] * x_shape[2], num_input_dims * x_shape[-1]))
	x = tf.unstack(x, x_shape[1] * x_shape[2], axis=1)

	#initial_cell_state = cell.zero_state(batch_size, tf.float32)
	#initial_cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, cell.state_size[i].c]), tf.zeros([batch_size, cell.state_size[i].h]))] * num_dims
	#cell_outputs, cell_state = tf.nn.static_rnn(cell, x, initial_state=initial_cell_state, dtype=tf.float32)
	cell_outputs, _ = tf.nn.static_rnn(cell, x, dtype=tf.float32)

	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
	cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
	# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
	cell_outputs = tf.transpose(cell_outputs, [2, 0, 1, 3])
	# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
	y_shape = cell_outputs.shape.as_list()
	cell_outputs = tf.reshape(cell_outputs, (-1, x_shape[1], x_shape[2], num_output_dims * y_shape[-1]))

	fc = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax)

	return fc

# Builds model.
print('Start building model...')
total_elapsed_time = time.time()
with tf.variable_scope('2d_grid_rnn', reuse=tf.AUTO_REUSE):
	y_pred = build_grid_cell_model(x_ph, num_dims, num_hidden_units, num_layers, keep_prob)

	y_pred_vec = tf.reshape(y_pred, [-1, num_classes])
	y_ph_vec = tf.reshape(y_ph, [-1, num_classes])

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_vec, labels=y_ph_vec))
	train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))

# Initializes variables.
initializer = tf.global_variables_initializer()
sess = tf.Session()
sess.run(initializer)

# Trains model.
print('Start training model...')
total_elapsed_time = time.time()
steps_per_epoch = math.ceil(num_examples / batch_size)
for epoch in range(1, num_epochs + 1):
	print('Epoch {}/{}'.format(epoch, num_epochs))

	#pdb.set_trace

	indices = np.arange(num_examples)
	np.random.shuffle(indices)

	for step in range(steps_per_epoch):
		start = step * batch_size
		end = start + batch_size
		batch_indices = indices[start:end]
		data_batch, label_batch = x[batch_indices,], y_gt[batch_indices,]

		sess.run(train_step, feed_dict={x_ph: data_batch, y_ph: label_batch})

		print('-', sep='', end='')
	print()
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))

# Evaluates model.
print('Start evaluating model...')
total_elapsed_time = time.time()
pred = y_pred.eval(session=sess, feed_dict={x_ph: x, y_ph: y_gt})
correct_prediction_count = np.count_nonzero(np.equal(np.argmax(pred, axis=-1), np.argmax(y_gt, axis=-1)))
print('\tAccurary = {} / {} = {}'.format(correct_prediction_count, y_gt.size, correct_prediction_count / y_gt.size))
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))

#%%-------------------------------------------------------------------

img_filename = 'D:/dataset/pattern_recognition/street1.jpg'

img = Image.open(img_filename)
img = np.asarray(img, dtype='uint8') / 255
#img = img.reshape((-1,) + img.shape)
img = img.reshape((-1,) + (img.shape[0] // 4, img.shape[1] // 4, img.shape[2]))
#img = np.random.randn(10, 100, 100, 3)
#img = np.random.rand(10, 100, 100, 3)

num_dims = 2
num_classes = 2
batch_size, img_height, img_width = img.shape[:3]
is_time_major = False

#%%-------------------------------------------------------------------
num_hidden_units = 32
keep_prob = 0.5

import grid_rnn_cell

def create_unit_cell(num_hidden_units, tied=False, non_recurrent_fn=None):
	#num_frequency_blocks = [1]
	#start_freqindex_list = [0]
	#end_freqindex_list = [1]
	#return tf.contrib.rnn.GridLSTMCell(num_hidden_units, num_frequency_blocks=num_frequency_blocks, start_freqindex_list=start_freqindex_list, end_freqindex_list=end_freqindex_list, reuse=True)

	# NOTICE [info] >> tf.contrib.grid_rnn.Grid2LSTMCell receives input data from 0-th dimension only.
	#return tf.contrib.grid_rnn.Grid2LSTMCell(num_hidden_units, tied, non_recurrent_fn)

	#def cell_fn(num_hidden_units):
	#	return tf.contrib.rnn.LSTMCell(num_units=num_hidden_units, forget_bias=1.0, use_peepholes=False)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=[0, 1], output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=cell_fn, non_recurrent_fn=non_recurrent_fn)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=[0, 1], output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=0, output_dims=[0, 1], priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)
	# NOTICE [info] >> 'input_dims=None & output_dims=None' means that there are no input and no output.
	#return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=None, output_dims=None, priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)
	return grid_rnn_cell.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=0, output_dims=0, priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)

# Defines a cell.
cell = create_unit_cell(num_hidden_units)
#cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=1.0, state_keep_prob=keep_prob)

#%%
total_elapsed_time = time.time()
cell_state = cell.zero_state(batch_size, tf.float32)
#cell_state = [(tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, cell.state_size[i].c]), tf.zeros([batch_size, cell.state_size[i].h]))) for i in range(num_dims)]
cell_outputs = []
#input_tensor = tf.convert_to_tensor(img, np.float32)
inp = tf.convert_to_tensor(img[:, 0, 0, :], np.float32)  # 1-dimensional input.
for hh in range(img_height):
	for ww in range(img_width):
		#inp = tf.convert_to_tensor(img[:, hh, ww, :], np.float32)  # 1-dimensional input.
		#inp = tf.convert_to_tensor(np.concatenate((img[:, hh, ww, :], img[:, hh, ww, :]), axis=1), np.float32)  # 2-dimensional input.
		#cell_output, cell_state = cell(inp, cell_state)
		#inp = input_tensor[:, hh, ww, :]  # 1-dimensional input.
		#inp = tf.concat((input_tensor[:, hh, ww, :], input_tensor[:, hh, ww, :]), axis=1)  # 2-dimensional input.
		#cell_output, cell_state = cell(inp, cell_state)
		#cell_outputs.append(cell_output)
		cell_output, _ = cell(inp, cell_state, scope='grid2lstm')
	print('.', end='')
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))
