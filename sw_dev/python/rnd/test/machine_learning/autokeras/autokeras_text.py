#!/usr/bin/env python

import time
import numpy as np
import tensorflow as tf
import autokeras as ak

# REF [site] >> https://autokeras.com/start/
def main():
	# Loads dataset.
	# FIXME [implement] >>
	"""
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.reshape(x_train.shape + (1,))
	x_test = x_test.reshape(x_test.shape + (1,))
	"""

	#--------------------
	clf = ak.TextClassifier(verbose=True)
		
	print('Fitting...')
	start_time = time.time()
	clf.fit(x_train, y_train, time_limit=12 * 60 * 60)  # time_limit in secs.
	print('\tElapsed time = {}'.format(time.time() - start_time))

	print('Final Fitting...')
	start_time = time.time()
	clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	print('Evaluating...')
	start_time = time.time()
	accuracy = clf.evaluate(x_test, y_test)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	print('Accuracy =', accuracy * 100)

	print('Predicting...')
	start_time = time.time()
	predictions = clf.predict(x_test)
	print('\tElapsed time = {}'.format(time.time() - start_time))

	print('Predictions =', predictions)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
