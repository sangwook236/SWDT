#!/usr/bin/env python

# REF [site] >>
#	https://www.tensorflow.org/guide/effective_tf2
#	https://www.tensorflow.org/guide/migrate
#	https://www.tensorflow.org/guide/upgrade

def tf1_compatibility():
	import tensorflow.compat.v1 as tf
	tf.disable_v2_behavior()

	a = tf.constant(2)
	b = tf.constant(3)
	c = a + b

	sess = tf.Session()
	print('c = a + b =', sess.run(c))

import tensorflow.compat.v2 as tf

def basic_operation():
	raise NotImplementedError

def main():
	#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # [0, 3].

	tf1_compatibility()

	#basic_operation()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
