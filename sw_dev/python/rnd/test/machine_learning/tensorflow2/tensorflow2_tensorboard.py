#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://www.tensorflow.org/tensorboard/

from __future__ import absolute_import, division, print_function, unicode_literals
import os, datetime
import tensorflow as tf

def create_model():
	return tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(10, activation='softmax')
	])

# REF [site] >> https://www.tensorflow.org/tensorboard/get_started
def get_started_keras_example():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Use TensorBoard with Keras Model.fit().

	model = create_model()
	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	os.makedirs(log_dir)
	os.makedirs(log_dir + '/train/plugins/profile')
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	model.fit(x=x_train, 
		y=y_train,
		epochs=5,
		validation_data=(x_test, y_test),
		callbacks=[tensorboard_callback])

	# TensorBoard:
	#	tensorboard --logdir logs/fit

# REF [site] >> https://www.tensorflow.org/tensorboard/get_started
def get_started_tf_example():
	# When training with methods such as tf.GradientTape(), use tf.summary to log the required information.

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

	train_dataset = train_dataset.shuffle(60000).batch(64)
	test_dataset = test_dataset.batch(64)

	# Choose loss and optimizer.
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam()

	# Define our metrics.
	train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
	test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

	# Define the training and test functions.
	@tf.function
	def train_step(model, optimizer, x_train, y_train):
		with tf.GradientTape() as tape:
			predictions = model(x_train, training=True)
			loss = loss_object(y_train, predictions)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

		train_loss(loss)
		train_accuracy(y_train, predictions)

	@tf.function
	def test_step(model, x_test, y_test):
		predictions = model(x_test)
		loss = loss_object(y_test, predictions)

		test_loss(loss)
		test_accuracy(y_test, predictions)

	# Set up summary writers to write the summaries to disk in a different logs directory.
	current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
	train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
	test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
	train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	test_summary_writer = tf.summary.create_file_writer(test_log_dir)

	# Start training.
	model = create_model()  # Reset our model.

	num_epochs = 5
	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	for epoch in range(num_epochs):
		for (x_train, y_train) in train_dataset:
			train_step(model, optimizer, x_train, y_train)
		with train_summary_writer.as_default():
			tf.summary.scalar('loss', train_loss.result(), step=epoch)
			tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

		for (x_test, y_test) in test_dataset:
			test_step(model, x_test, y_test)
		with test_summary_writer.as_default():
			tf.summary.scalar('loss', test_loss.result(), step=epoch)
			tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

		print (template.format(epoch + 1,
			train_loss.result(), 
			train_accuracy.result() * 100,
			test_loss.result(), 
			test_accuracy.result() * 100))

		# Reset metrics every epoch.
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

	# TensorBoard:
	#	tensorboard --logdir logs/gradient_tape

def main():
	#get_started_keras_example()
	get_started_tf_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
