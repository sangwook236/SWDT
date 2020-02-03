#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class MyModel(tf.keras.Model):
	def __init__(self):
		super(MyModel, self).__init__()
		self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
		self.flatten = tf.keras.layers.Flatten()
		self.d1 = tf.keras.layers.Dense(128, activation='relu')
		self.d2 = tf.keras.layers.Dense(10, activation='softmax')

	def call(self, x):
		x = self.conv1(x)
		x = self.flatten(x)
		x = self.d1(x)
		return self.d2(x)

# REF [site] >> https://www.tensorflow.org/tutorials/quickstart/advanced
def mnist_example():
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	# Add a channels dimension.
	x_train = x_train[..., tf.newaxis]
	x_test = x_test[..., tf.newaxis]

	batch_size = 32
	train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
	test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

	#--------------------
	# Create an instance of the model.
	model = MyModel()

	#--------------------
	# Choose an optimizer and loss function for training.
	loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.Adam()

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

	test_loss = tf.keras.metrics.Mean(name='test_loss')
	test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

	@tf.function
	def train_step(images, labels):
		with tf.GradientTape() as tape:
			predictions = model(images)
			loss = loss_object(labels, predictions)
		variables = model.trainable_variables
		gradients = tape.gradient(loss, variables)
		"""
		# Gradient clipping.
		max_gradient_norm = 5
		gradients = list(map(lambda grad: (tf.clip_by_norm(grad, clip_norm=max_gradient_norm)), gradients))
		#gradients = list(map(lambda grad: (tf.clip_by_value(grad, clip_value_min=min_clip_val, clip_value_max=max_clip_val)), gradients))
		"""
		optimizer.apply_gradients(zip(gradients, variables))

		train_loss(loss)
		train_accuracy(labels, predictions)

	@tf.function
	def test_step(images, labels):
		predictions = model(images)
		loss = loss_object(labels, predictions)

		test_loss(loss)
		test_accuracy(labels, predictions)

	#--------------------
	# Train.
	num_epochs = 5

	template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
	for epoch in range(num_epochs):
		for images, labels in train_ds:
			train_step(images, labels)

		for test_images, test_labels in test_ds:
			test_step(test_images, test_labels)

		print(template.format(epoch + 1,
			train_loss.result(),
			train_accuracy.result() * 100,
			test_loss.result(),
			test_accuracy.result() * 100)
		)

		# Reset the metrics for the next epoch.
		train_loss.reset_states()
		train_accuracy.reset_states()
		test_loss.reset_states()
		test_accuracy.reset_states()

def main():
	mnist_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
