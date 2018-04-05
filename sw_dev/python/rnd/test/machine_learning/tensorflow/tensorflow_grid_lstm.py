import tensorflow as tf
from PIL import Image
import numpy as np
import time

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

#%%
sess = tf.Session()

#%%

# REF [site] >> https://github.com/KendallWeihe/Tensorflow-GridRNNCell-example/blob/master/GridLSTM_ex.py

num_dims = 3
num_classes = 2
num_hidden_units = 128
keep_prob = 0.8

input_shape = (20, 30, 4, 3)  # (height, width, depth, channels).
ouput_shape = input_shape[:3] + (num_classes,)

x_ph = tf.placeholder(tf.float32, input_shape)
y_ph = tf.placeholder(tf.float32, ouput_shape)

# Generates random data.
x = np.random.rand(input_shape[0], input_shape[1], input_shape[2], input_shape[3])
y_gt = np.random.rand(ouput_shape[0], ouput_shape[1], ouput_shape[2], ouput_shape[3])

def run_grid_cell(x, num_dims, num_hidden_units, keep_prob):
	#import pdb
	#pdb.set_trace()

	cell = tf.contrib.grid_rnn.GridRNNCell(num_hidden_units, num_dims=num_dims)
	cell = tf.contrib.rnn.MultiRNNCell([cell] * num_hidden_units)
	cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

	x = tf.unstack(x, num_time_steps, axis=1)
	outputs, state = tf.nn.static_rnn(cell, x, dtype=tf.float32)
	outputs = tf.stack(outputs, axis=1)
	return outputs

with tf.variable_scope('gridrnn', reuse=tf.AUTO_REUSE):
	pred = run_grid_cell(x_ph, num_dims, num_hidden_units, keep_prob)
