import tensorflow as tf
from PIL import Image
import numpy as np
import math
import time
#import matplotlib.pyplot as plt

#%%------------------------------------------------------------------

def to_one_hot_encoding(label_indexes, num_classes):
	return np.eye(num_classes)[label_indexes].reshape(label_indexes.shape + (-1,))

def create_grid_cell(num_hidden_units, num_dims, input_dims, output_dims, tied=False, non_recurrent_fn=None):
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
	return tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims, input_dims=input_dims, output_dims=output_dims, priority_dims=None, non_recurrent_dims=None, tied=tied, cell_fn=None, non_recurrent_fn=non_recurrent_fn)

#%%------------------------------------------------------------------
# An example of 2D GridRNN.

def create_grid_rnn_model(x, num_dims, num_hidden_units, num_layers, keep_prob):
	#import pdb
	#pdb.set_trace()

	input_dims = [0] #[0, 1]
	output_dims = [0] #[0, 1]
	num_input_dims = len(input_dims)
	num_output_dims = len(output_dims)

	cell = create_grid_cell(num_hidden_units, num_dims, input_dims, output_dims)
	if num_layers is not None and num_layers > 1:
		cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

	# Input of 1 dimension = an 1-vector of the same length of the feature vector.
	# Input of 2 dimensions = an 1-vector of twice the length of the feature vector.
	# Input of n dimensions = an 1-vector of an n-fold length of the feature vector. (num_input_dims * features).

	# A tensor of shape (samples, height, width, features) -> a list of 'height * width' tensors of shape (samples, features).
	x_shape = x.shape
	# (samples, height, width, features) -(?)-> (samples, height * width, num_input_dims, features) or (samples, height * width, num_input_dims * features).
	x = tf.reshape(x, (-1, x_shape[1] * x_shape[2], num_input_dims * x_shape[-1]))
	x = tf.unstack(x, x_shape[1] * x_shape[2], axis=1)

	#initial_cell_state = cell.zero_state(batch_size, tf.float32)
	#batch_size_ph = tf.placeholder(tf.int32, (1,))  # ???
	#initial_cell_state = cell.zero_state(batch_size_ph, tf.float32)
	#initial_cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, cell.state_size[i].c]), tf.zeros([batch_size, cell.state_size[i].h])) for i in range(num_dims)]
	#initial_cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size_ph, cell.state_size[i].c]), tf.zeros([batch_size_ph, cell.state_size[i].h])) for i in range(num_dims)]
	#cell_outputs, cell_state = tf.nn.static_rnn(cell, x, initial_state=initial_cell_state, dtype=tf.float32)
	cell_outputs, _ = tf.nn.static_rnn(cell, x, dtype=tf.float32)

	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
	cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
	# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
	cell_outputs = tf.transpose(cell_outputs, [2, 0, 1, 3])
	# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
	y_shape = cell_outputs.shape
	cell_outputs = tf.reshape(cell_outputs, (-1, x_shape[1], x_shape[2], num_output_dims * y_shape[-1]))

	fc = tf.layers.dense(cell_outputs, num_classes, activation=tf.nn.softmax)

	return fc

# Prepares data.
print('Preparing data...')

num_dims = 2
num_classes = 2
num_examples = 10
batch_size = 4
num_epochs = 1000
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

# Generates random data.
x = np.random.rand(num_examples, input_shape[1], input_shape[2], input_shape[3])
y_gt = np.random.rand(num_examples, ouput_shape[1], ouput_shape[2], ouput_shape[3])
y_gt = to_one_hot_encoding(np.argmax(y_gt, axis=-1), num_classes)

# Builds model.
print('Building model...')
total_elapsed_time = time.time()
with tf.variable_scope('2d_grid_rnn', reuse=tf.AUTO_REUSE):
	y_pred = create_grid_rnn_model(x_ph, num_dims, num_hidden_units, num_layers, keep_prob)

	y_pred_vec = tf.reshape(y_pred, [-1, num_classes])
	y_ph_vec = tf.reshape(y_ph, [-1, num_classes])

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_vec, labels=y_ph_vec))
	train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))

# Initializes variables.
initializer = tf.global_variables_initializer()
sess = tf.Session()
sess.run(initializer)

# Trains model.
print('Training model...')
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
print('Evaluating model...')
total_elapsed_time = time.time()
pred = y_pred.eval(session=sess, feed_dict={x_ph: x, y_ph: y_gt})
correct_prediction_count = np.count_nonzero(np.equal(np.argmax(pred, axis=-1), np.argmax(y_gt, axis=-1)))
print('\tAccurary = {} / {} = {}'.format(correct_prediction_count, y_gt.size, correct_prediction_count / y_gt.size))
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))

#%%-------------------------------------------------------------------

def pad_image(img, target_height, target_width):
	if 2 == img.ndim:
		height, width = img.shape
	elif 3 == img.ndim:
		height, width, _ = img.shape
	else:
		assert 2 == img.ndim or 3 == img.ndim, 'The dimension of an image is not proper.'

	left_margin = (target_width - width) // 2
	right_margin = target_width - width - left_margin
	#top_margin = (target_height - height) // 2
	#bottom_margin = target_height - height - top_margin
	top_margin = target_height - height
	bottom_margin = target_height - height - top_margin
	if 2 == img.ndim:
		return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin)), 'edge')
		#return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin)), 'constant', constant_values=(0, 0))
	else:
		return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin), (0, 0)), 'edge')
		#return np.pad(img, ((top_margin, bottom_margin), (left_margin, right_margin), (0, 0)), 'constant', constant_values=(0, 0))

# Traverses pixels: '(left -> right) + (top -> bottom)' or '(top -> bottom) + (left -> right)'.
def traverse_pixel(grid_cell, input_tensor, is_row_major=True):
	# Traverses input tensor in row-major order: (left -> right) + (top -> bottom).
	def traverse_in_row_major_order(grid_cell, input_tensor, num_input_dims=1, num_output_dims=1):
		# A tensor of shape (samples, height, width, features) -> a list of 'height * width' tensors of shape (samples, features).
		batch_size, height, width = input_tensor.shape.as_list()[0:3]
		# (samples, height, width, features) -(?)-> (samples, height * width, num_input_dims, features) or (samples, height * width, num_input_dims * features).
		input_tensor = tf.reshape(input_tensor, (-1, height * width, num_input_dims * input_tensor.shape[-1]))
		input_tensor = tf.unstack(input_tensor, height * width, axis=1)

		cell_outputs, cell_state = tf.nn.static_rnn(grid_cell, input_tensor, dtype=tf.float32)

		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
		cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
		# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
		cell_outputs = tf.transpose(cell_outputs, perm=(2, 0, 1, 3))
		# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
		cell_outputs = tf.reshape(cell_outputs, (-1, height, width, num_output_dims * cell_outputs.shape[-1]))
		return cell_outputs, cell_state

	# Traverses input tenso in column-major order: (top -> bottom) + (left -> right).
	def traverse_in_column_major_order(grid_cell, input_tensor, num_input_dims=1, num_output_dims=1):
		input_tensor = tf.transpose(input_tensor, perm=(0, 2, 1, 3))
		cell_outputs, cell_state = traverse_in_row_major_order(grid_cell, input_tensor, num_input_dims, num_output_dims)
		cell_outputs = tf.transpose(cell_outputs, perm=(0, 2, 1, 3))
		return cell_outputs, cell_state

	"""
	# Traverses input tenso in row-major order: (left -> right) + (top -> bottom).
	def traverse_in_row_major_order(grid_cell, input_tensor):
		batch_size, height, width = input_tensor.shape.as_list()[0:3]
		if batch_size is None:
			batch_size_ph = tf.placeholder(tf.int32, (1,))  # ???
			cell_state = grid_cell.zero_state(batch_size_ph, tf.float32)
			#cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size_ph, grid_cell.state_size[i].c]), tf.zeros([batch_size_ph, grid_cell.state_size[i].h])) for i in range(num_dims)]
		else:
			cell_state = grid_cell.zero_state(batch_size, tf.float32)
			#cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, grid_cell.state_size[i].c]), tf.zeros([batch_size, grid_cell.state_size[i].h])) for i in range(num_dims)]
		cell_outputs = []
		for hh in range(height):
			for ww in range(width):
				inp = input_tensor[:, hh, ww, :]  # 1-dimensional input.
				#inp = tf.concat((input_tensor[:, hh, ww, :], input_tensor[:, hh, ww, :]), axis=1)  # 2-dimensional input.
				cell_output, cell_state = grid_cell(inp, cell_state)
				cell_outputs.append(cell_output)

		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
		cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
		# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
		cell_outputs = tf.transpose(cell_outputs, perm=(2, 0, 1, 3))
		# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
		cell_outputs = tf.reshape(cell_outputs, (-1, height, width, num_output_dims * cell_outputs.shape[-1]))
		return cell_outputs, cell_state

	# Traverses input tenso in column-major order: (top -> bottom) + (left -> right).
	def traverse_in_column_major_order(grid_cell, input_tensor):
		batch_size, height, width = input_tensor.shape.as_list()[0:3]
		if batch_size is None:
			batch_size_ph = tf.placeholder(tf.int32, (1,))  # ???
			cell_state = grid_cell.zero_state(batch_size_ph, tf.float32)
			#cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size_ph, grid_cell.state_size[i].c]), tf.zeros([batch_size_ph, grid_cell.state_size[i].h])) for i in range(num_dims)]
		else:
			cell_state = grid_cell.zero_state(batch_size, tf.float32)
			#cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, grid_cell.state_size[i].c]), tf.zeros([batch_size, grid_cell.state_size[i].h])) for i in range(num_dims)]
		cell_outputs = []
		for ww in range(width):
			for hh in range(height):
				inp = input_tensor[:, hh, ww, :]  # 1-dimensional input.
				#inp = tf.concat((input_tensor[:, hh, ww, :], input_tensor[:, hh, ww, :]), axis=1)  # 2-dimensional input.
				cell_output, cell_state = grid_cell(inp, cell_state)
				cell_outputs.append(cell_output)

		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
		# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
		cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
		# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
		cell_outputs = tf.transpose(cell_outputs, perm=(2, 0, 1, 3))
		# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
		cell_outputs = tf.reshape(cell_outputs, (-1, height, width, num_output_dims * cell_outputs.shape[-1]))
		cell_outputs = tf.transpose(cell_outputs, perm=(0, 2, 1, 3))
		return cell_outputs, cell_state
	"""

	num_input_dims, num_output_dims = 1, 1
	if is_row_major:
		return traverse_in_row_major_order(grid_cell, input_tensor, num_input_dims, num_output_dims)
	else:
		return traverse_in_column_major_order(grid_cell, input_tensor, num_input_dims, num_output_dims)

# Traverses columns: left -> right.
def traverse_column(grid_cell, input_tensor, batch_size, num_output_dims=1):
	height, width = input_tensor.shape.as_list()[1:3]
	initial_cell_state = grid_cell.zero_state(batch_size, tf.float32)  # A tuple of 'num_recurrent_dims' LSTMStateTuple's.
	cell_states = [initial_cell_state] * height
	#cell_states = [[tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, grid_cell.state_size[i].c]), tf.zeros([batch_size, grid_cell.state_size[i].h])) for i in range(num_dims)]] * height
	cell_outputs = []
	for ww in range(width):
		prev_col_state = initial_cell_state[0]
		for hh in range(height):
			inp = input_tensor[:, hh, ww, :]  # 1-dimensional input.
			#inp = tf.concat((input_tensor[:, hh, ww, :], input_tensor[:, hh, ww, :]), axis=1)  # 2-dimensional input.
			prev_row_state = cell_states[hh][1]
			cell_output, cell_states[hh] = grid_cell(inp, (prev_col_state, prev_row_state))
			prev_col_state = cell_states[hh][0]
			cell_outputs.append(cell_output)

	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
	cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
	# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
	cell_outputs = tf.transpose(cell_outputs, perm=(2, 0, 1, 3))
	# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
	cell_outputs = tf.reshape(cell_outputs, (-1, height, width, num_output_dims * cell_outputs.shape[-1]))
	return cell_outputs, cell_states

# Traverses rows: top -> bottom.
def traverse_row(grid_cell, input_tensor, batch_size, num_output_dims=1):
	height, width = input_tensor.shape.as_list()[1:3]
	initial_cell_state = grid_cell.zero_state(batch_size, tf.float32)  # A tuple of 'num_recurrent_dims' LSTMStateTuple's.
	cell_states = [initial_cell_state] * width
	#cell_states = [[tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, grid_cell.state_size[i].c]), tf.zeros([batch_size, grid_cell.state_size[i].h])) for i in range(num_dims)]] * width
	cell_outputs = []
	for hh in range(height):
		prev_row_state = initial_cell_state[1]
		for ww in range(width):
			inp = input_tensor[:, hh, ww, :]  # 1-dimensional input.
			#inp = tf.concat((input_tensor[:, hh, ww, :], input_tensor[:, hh, ww, :]), axis=1)  # 2-dimensional input.
			prev_col_state = cell_states[ww][0]
			cell_output, cell_states[ww] = grid_cell(inp, (prev_col_state, prev_row_state))
			prev_row_state = cell_states[ww][1]
			cell_outputs.append(cell_output)

	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> a tensor of shape (samples, height, width, num_output_dims, features) or (samples, height, width, num_output_dims * features).
	# A list of 'height * width' tuples of 'num_output_dims' tensors of shape (samples, features) -> (height * width, num_output_dims, samples, features).
	cell_outputs = tf.stack(cell_outputs, axis=0)  # Output of each cell is n-tuple. Here n = num_output_dims.
	# (height * width, num_output_dims, samples, features) -> (samples, height * width, num_output_dims, features).
	cell_outputs = tf.transpose(cell_outputs, perm=(2, 0, 1, 3))
	# (samples, height * width, num_output_dims, features) -> (samples, height, width, num_output_dims * features).
	cell_outputs = tf.reshape(cell_outputs, (-1, height, width, num_output_dims * cell_outputs.shape[-1]))
	return cell_outputs, cell_states

def create_grid_encdec_model(x, num_dims, num_hidden_units, num_layers, keep_prob):
	#import pdb
	#pdb.set_trace()

	input_dims = [0] #[0, 1]
	output_dims = [0] #[0, 1]
	num_input_dims = len(input_dims)
	num_output_dims = len(output_dims)

	def create_unit_cell():
		cell = create_grid_cell(num_hidden_units, num_dims, input_dims, output_dims)
		if num_layers is not None and num_layers > 1:
			cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
		cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
		return cell

	# (top left, top right, bottom left, bottom right).
	enc_cells = create_unit_cell(), create_unit_cell(), create_unit_cell(), create_unit_cell()
	# (top left, top right, bottom left, bottom right).
	dec_cells = create_unit_cell(), create_unit_cell(), create_unit_cell(), create_unit_cell()

	# Input of 1 dimension = an 1-vector of the same length of the feature vector.
	# Input of 2 dimensions = an 1-vector of twice the length of the feature vector.
	# Input of n dimensions = an 1-vector of an n-fold length of the feature vector. (num_input_dims * features).

	# A tensor of shape (samples, height, width, features) -> a list of 'height * width' tensors of shape (samples, features).
	x_shape = x.shape.as_list()
	# (samples, height, width, features) -(?)-> (samples, height * width, num_input_dims, features) or (samples, height * width, num_input_dims * features).
	x = tf.reshape(x, (-1, x_shape[1] * x_shape[2], num_input_dims * x_shape[-1]))
	x = tf.unstack(x, x_shape[1] * x_shape[2], axis=1)

	#initial_cell_state = cell.zero_state(batch_size, tf.float32)
	#batch_size_ph = tf.placeholder(tf.int32, (1,))
	#initial_cell_state = cell.zero_state(batch_size_ph, tf.float32)
	#initial_cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, cell.state_size[i].c]), tf.zeros([batch_size, cell.state_size[i].h])) for i in range(num_dims)]
	#initial_cell_state = [tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size_ph, cell.state_size[i].c]), tf.zeros([batch_size_ph, cell.state_size[i].h])) for i in range(num_dims)]
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

# Prepares data.
print('Preparing data...')

rgb_img_filename = 'D:/dataset/phenotyping/RDA/all_plants_mask/plant/1.5xN_1(16.04.22) - ang0.png'
mask_img_filename = 'D:/dataset/phenotyping/RDA/all_plants_mask/mask/1.5xN_1(16.04.22) - ang0_mask00.png'
#mask_img_filename = 'D:/dataset/phenotyping/RDA/all_plants_mask/mask/1.5xN_1(16.04.22) - ang0_mask01.png'
#mask_img_filename = 'D:/dataset/phenotyping/RDA/all_plants_mask/mask/1.5xN_1(16.04.22) - ang0_mask02.png'

rgb_img = Image.open(rgb_img_filename)
rgb_img = np.asarray(rgb_img, dtype='uint8') / 255
mask_img = Image.open(mask_img_filename)
mask_img = np.asarray(mask_img, dtype='uint8') / 255

img_max_len = 300
rgb_img = pad_image(rgb_img, img_max_len, img_max_len)
mask_img = pad_image(mask_img, img_max_len, img_max_len)

rgb_img = rgb_img.reshape((-1,) + rgb_img.shape)
mask_img = mask_img.reshape((-1,) + mask_img.shape)

num_dims = 2
num_classes = 2
num_examples = 10
batch_size = 4
num_epochs = 1000
num_hidden_units = 16
num_layers = None
keep_prob = 0.8

input_shape = (None,) + rgb_img.shape[1:]  # (samples, height, width, features(channels)).
ouput_shape = input_shape[:3] + (num_classes,)

x_ph = tf.placeholder(tf.float32, input_shape)
y_ph = tf.placeholder(tf.float32, ouput_shape)

with tf.variable_scope('2d_grid_encdec', reuse=tf.AUTO_REUSE):
	input_dims = [0] #[0, 1]
	output_dims = [0] #[0, 1]
	cell = create_grid_cell(num_hidden_units, num_dims, input_dims, output_dims)
	cell_outputs, cell_states = traverse_column(cell, x_ph, batch_size)
	#cell_outputs, cell_states = traverse_row(cell, x_ph, batch_size)

# Builds model.
print('Building model...')
total_elapsed_time = time.time()
with tf.variable_scope('2d_grid_encdec', reuse=tf.AUTO_REUSE):
	y_pred = create_grid_encdec_model(x_ph, num_dims, num_hidden_units, num_layers, keep_prob)

	y_pred_vec = tf.reshape(y_pred, [-1, num_classes])
	y_ph_vec = tf.reshape(y_ph, [-1, num_classes])

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_vec, labels=y_ph_vec))
	train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
print('\tTotal time = {}'.format(time.time() - total_elapsed_time))
