#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf

# Define a simple sequential model
def create_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy'])

	return model

# REF [site] >> https://www.tensorflow.org/tutorials/keras/save_and_load
def keras_example():
	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

	train_labels = train_labels[:1000]
	test_labels = test_labels[:1000]

	train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
	test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

	#----------------------------------------
	# Save checkpoints during training.
	# Checkpoint callback usage.

	# Create a basic model instance.
	model = create_model()

	# Display the model's architecture.
	model.summary()

	# Create a callback that saves the model's weights.
	checkpoint_filepath = 'training_1/ckpt'
	#os.makedirs(checkpoint_filepath + '/variables')  # When save_weights_only = False.
	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		save_weights_only=True,
		verbose=1)
	callbacks = [checkpoint_callback]

	# Train the model with the new callback.
	model.fit(train_images, 
		train_labels,  
		epochs=10,
		validation_data=(test_images, test_labels),
		callbacks=callbacks)  # Pass callback to training.

	# This may generate warnings related to saving the state of the optimizer.
	# These warnings (and similar warnings throughout this notebook)
	# are in place to discourage outdated usage, and can be ignored.

	#--------------------
	# Create a basic model instance.
	model = create_model()

	# Evaluate the model.
	loss, acc = model.evaluate(test_images, test_labels, verbose=2)
	print('Untrained model, accuracy: {:5.2f}%'.format(100 * acc))

	# Loads the weights.
	model.load_weights(checkpoint_filepath)

	# Re-evaluate the model.
	loss, acc = model.evaluate(test_images, test_labels, verbose=2)
	print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

	#----------------------------------------
	# Checkpoint callback options.

	# Include the epoch in the file name (uses 'str.format').
	checkpoint_filepath = 'training_2/ckpt.{epoch:04d}'
	#checkpoint_filepath = 'training_2/ckpt.{epoch:04d}-{val_loss:.5f}'

	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_filepath,
		#monitor='val_loss',
		verbose=1,
		save_best_only=True,
		save_weights_only=True,
		mode='auto',
		period=5)
	callbacks = [checkpoint_callback]

	# Create a new model instance.
	model = create_model()

	# Save the weights using the 'checkpoint_filepath' format.
	model.save_weights(checkpoint_filepath.format(epoch=0))

	# Train the model with the new callback.
	model.fit(train_images, 
		train_labels,
		epochs=50, 
		callbacks=callbacks,
		validation_data=(test_images, test_labels),
		verbose=0)

	#--------------------
	checkpoint_dir_path = os.path.dirname(checkpoint_filepath)
	latest_checkpoint_filepath = tf.train.latest_checkpoint(checkpoint_dir_path)
	print('Latest checkpoint filepath = {}.'.format(latest_checkpoint_filepath))

	# Create a new model instance.
	model = create_model()

	# Load the previously saved weights.
	model.load_weights(latest_checkpoint_filepath)

	# Re-evaluate the model.
	loss, acc = model.evaluate(test_images, test_labels, verbose=2)
	print('Restored model, accuracy: {:5.2f}%.'.format(100 * acc))

	#----------------------------------------
	# Manually save weights.

	# Save the weights.
	model.save_weights('./checkpoints/my_ckpt')

	# Create a new model instance.
	model = create_model()

	# Restore the weights.
	model.load_weights('./checkpoints/my_ckpt')

	# Evaluate the model.
	loss,acc = model.evaluate(test_images, test_labels, verbose=2)
	print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

class MyModel(tf.keras.Model):
	"""A simple linear model."""

	def __init__(self):
		super(MyModel, self).__init__()
		self.l1 = tf.keras.layers.Dense(5)

	def call(self, x):
		return self.l1(x)

# REF [site] >> https://www.tensorflow.org/guide/checkpoint
def checkpoint_example():
	def toy_dataset():
		inputs = tf.range(10.)[:, None]
		labels = inputs * 5. + tf.range(5.)[None, :]
		return tf.data.Dataset.from_tensor_slices(dict(x=inputs, y=labels)).repeat(10).batch(2)

	def train_step(model, example, optimizer):
		"""Trains 'model' on 'example' using 'optimizer'."""
		with tf.GradientTape() as tape:
			output = model(example['x'])
			loss = tf.reduce_mean(tf.abs(output - example['y']))
		variables = model.trainable_variables
		gradients = tape.gradient(loss, variables)
		optimizer.apply_gradients(zip(gradients, variables))
		return loss

	#--------------------
	model = MyModel()

	optimizer = tf.keras.optimizers.Adam(0.1)

	# Create the checkpoint objects.
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
	manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

	# Train and checkpoint the model.
	ckpt.restore(manager.latest_checkpoint)
	if manager.latest_checkpoint:
		print('Restored from {}.'.format(manager.latest_checkpoint))
	else:
		print('Initializing from scratch.')

	for example in toy_dataset():
		loss = train_step(model, example, optimizer)
		ckpt.step.assign_add(1)
		if int(ckpt.step) % 10 == 0:
			save_path = manager.save()
			print('Saved checkpoint for step {}: {}.'.format(int(ckpt.step), save_path))
			print('Loss: {:1.2f}.'.format(loss.numpy()))

# REF [site] >> https://www.tensorflow.org/guide/saved_model
def saved_model_example():
	raise NotImplementedError

def main():
	keras_example()
	#checkpoint_example()
	#saved_model_example()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
