#!/usr/bin/env python

# REF [site] >> https://github.com/baidu-research/warp-ctc

import numpy as np
import tensorflow as tf
import warpctc_tensorflow

# REF [site] >> https://github.com/baidu-research/warp-ctc/blob/master/tests/test_cpu.cpp
def simple_toy_example_1():
	#num_time_steps = 5
	#num_batches = 2
	#alphabet_size = 6  # Feature size.
	blank_label = 5

	activations = np.array([
		[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
		 [0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508]],
		[[0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
		 [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549]],
		[[0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
		 [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456]],
		[[0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
		 [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345]],
		[[0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107],
		 [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]
	])
	activations = np.log(activations)  # ???
	activation_lens = np.array([5, 5])
	labels = np.array([
		0, 1, 2, 1, 0,
		0, 1, 1, 0
	])
	label_lens = np.array([5, 4])

	# Expected CTC = [3.3421143650988143, 5.42262].
	ctc_costs = warpctc_tensorflow.ctc(activations, labels, label_lens, activation_lens, blank_label=blank_label)

	with tf.Session() as sess:
		costs = sess.run(ctc_costs)

	print('CTC costs =', costs)

def simple_toy_example_2():
	#num_time_steps = 2
	#num_batches = 3
	#alphabet_size = 5  # Feature size.
	blank_label = 0

	probs = np.array([
		[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]],
		[[0.1, 0.1, 0.1, 0.6, 0.1], [0.1, 0.1, 0.1, 0.1, 0.6]],
		[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.6, 0.1]]
	])
	probs = np.transpose(probs, (1, 0, 2))  # (batches, time-steps, features) -> (time-steps, batches, features).
	#probs = np.log(probs)  # ???
	prob_lens = np.array([2, 2, 2])
	#labels = np.array([[1, 2], [3, 4], [1, 3]])  # InvalidArgumentError (see above for traceback): flat_labels is not a vector.
	labels = np.array([
		1, 2,
		3, 4,
		1, 3
	])
	label_lens = np.array([2, 2, 2])

	ctc_costs = warpctc_tensorflow.ctc(probs, labels, label_lens, prob_lens, blank_label=blank_label)

	with tf.Session() as sess:
		costs = sess.run(ctc_costs)

	print('CTC costs =', costs)

"""
ctc(activations, flat_labels, label_lengths, input_lengths, blank_label=0)
	Computes the CTC loss between a sequence of activations and a ground truth labeling.

Inputs:
	activations: A 3-D Tensor of floats.
		The dimensions should be (t, n, a), where t is the time index, n is the minibatch index, and a indexes over activations for each symbol in the alphabet.
	flat_labels: A 1-D Tensor of ints, a concatenation of all the labels for the minibatch.
	label_lengths: A 1-D Tensor of ints, the length of each label for each example in the minibatch.
	input_lengths: A 1-D Tensor of ints, the number of time steps for each sequence in the minibatch.
	blank_label: int, the label value/index that the CTC calculation should use as the blank label.
Returns:
	1-D float Tensor, the cost of each example in the minibatch (as negative log probabilities).

This class performs the softmax operation internally.
The label reserved for the blank symbol should be label 0.
"""

def main():
	simple_toy_example_1()
	simple_toy_example_2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
