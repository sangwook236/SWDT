#!/usr/bin/env python

# REF [site] >> https://www.tensorflow.org/guide/eager

import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

def eager_execution_test():
	#--------------------
	# Eager execution provides an imperative interface to TensorFlow.
	# With eager execution enabled, TensorFlow functions execute operations immediately (as opposed to adding to a graph to be executed later in a tf.compat.v1.Session) and return concrete values (as opposed to symbolic references to a node in a computational graph).
	tf.enable_eager_execution()

	print('6 + 7 = ', tf.multiply(6, 7).numpy())

def main():
	eager_execution_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
