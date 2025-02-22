#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [paper] >> "Dynamic Routing Between Capsules", arXiv 2017.

from __future__ import print_function
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations
from tensorflow.keras import utils
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# The squashing function.
# We use 0.5 instead of 1 in Hinton's paper.
# If 1, the norm of vector will be zoomed out.
# If 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
	s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
	scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
	return scale * x

# Define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
	ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
	return ex / K.sum(ex, axis=axis, keepdims=True)

# Define the margin loss like hinge loss.
def margin_loss(y_true, y_pred):
	lamb, margin = 0.5, 0.1
	return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) +
		lamb * (1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)

class Capsule(Layer):
	"""A Capsule Implement with Pure Keras.
	There are two vesions of Capsule.
	One is like dense layer (for the fixed-shape input),
	and the other is like timedistributed dense (for various length input).

	The input shape of Capsule must be (batch_size, input_num_capsule, input_dim_capsule)
	and the output shape is (batch_size, num_capsule, dim_capsule)

	Capsule Implement is from https://github.com/bojone/Capsule/
	Capsule Paper: https://arxiv.org/abs/1710.09829
	"""

    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
		super(Capsule, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.routings = routings
		self.share_weights = share_weights
		if activation == 'squash':
			self.activation = squash
		else:
			self.activation = activations.get(activation)

	def build(self, input_shape):
		input_dim_capsule = input_shape.as_list()[-1]
		if self.share_weights:
			self.kernel = self.add_weight(
				name='capsule_kernel',
				shape=(1, input_dim_capsule, self.num_capsule * self.dim_capsule),
				initializer='glorot_uniform',
				trainable=True)
		else:
			input_num_capsule = input_shape[-2]
			self.kernel = self.add_weight(
				name='capsule_kernel',
				shape=(input_num_capsule, input_dim_capsule, self.num_capsule * self.dim_capsule),
				initializer='glorot_uniform',
				trainable=True)

    def call(self, inputs):
		"""Following the routing algorithm from Hinton's paper,
		but replace b = b + <u,v> with b = <u,v>.

		This change can improve the feature representation of Capsule.

		However, you can replace
			b = K.batch_dot(outputs, hat_inputs, [2, 3])
		with
			b += K.batch_dot(outputs, hat_inputs, [2, 3])
		to realize a standard routing.
		"""

		if self.share_weights:
			hat_inputs = K.conv1d(inputs, self.kernel)
		else:
			hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

		batch_size = K.shape(inputs)[0]
		input_num_capsule = K.shape(inputs)[1]
		hat_inputs = K.reshape(hat_inputs, (batch_size, input_num_capsule, self.num_capsule, self.dim_capsule))
		hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

		b = K.zeros_like(hat_inputs[:, :, :, 0])
		for i in range(self.routings):
			c = softmax(b, 1)
			o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
			if i < self.routings - 1:
				b = K.batch_dot(o, hat_inputs, [2, 3])
				if K.backend() == 'theano':
					o = K.sum(o, axis=1)

		return o

	def compute_output_shape(self, input_shape):
		return (None, self.num_capsule, self.dim_capsule)

# REF [site] >>
#	https://keras.io/examples/cifar10_cnn_capsule/
#	https://github.com/bojone/Capsule/
def cifar10_cnn_capsule_network_example():
	batch_size = 128
	num_classes = 10
	epochs = 100
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	y_train = utils.to_categorical(y_train, num_classes)
	y_test = utils.to_categorical(y_test, num_classes)

	# A common Conv2D model.
	input_image = Input(shape=(None, None, 3))
	x = Conv2D(64, (3, 3), activation='relu')(input_image)
	x = Conv2D(64, (3, 3), activation='relu')(x)
	x = AveragePooling2D((2, 2))(x)
	x = Conv2D(128, (3, 3), activation='relu')(x)
	x = Conv2D(128, (3, 3), activation='relu')(x)

	"""Now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
	then connect a Capsule layer.

	The output of final model is the lengths of 10 Capsule, whose dim=16.

	The length of Capsule is the proba,
	so the problem becomes a 10 two-classification problem.
	"""

	x = Reshape((-1, 128))(x)
	capsule = Capsule(10, 16, 3, True)(x)
	output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
	model = Model(inputs=input_image, outputs=output)

	# We use a margin loss.
	model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy'])
	model.summary()

	# We can compare the performance with or without data augmentation.
	data_augmentation = True

	if not data_augmentation:
		print('Not using data augmentation.')
		model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
	else:
		print('Using real-time data augmentation.')
		# This will do preprocessing and realtime data augmentation:
		datagen = ImageDataGenerator(
			featurewise_center=False,  # Set input mean to 0 over the dataset.
			samplewise_center=False,  # Set each sample mean to 0.
			featurewise_std_normalization=False,  # Divide inputs by dataset std.
			samplewise_std_normalization=False,  # Divide each input by its std.
			zca_whitening=False,  # Apply ZCA whitening.
			zca_epsilon=1e-06,  # Epsilon for ZCA whitening.
			rotation_range=0,  # Randomly rotate images in 0 to 180 degrees.
			width_shift_range=0.1,  # Randomly shift images horizontally.
			height_shift_range=0.1,  # Randomly shift images vertically.
			shear_range=0.,  # Set range for random shear.
			zoom_range=0.,  # Set range for random zoom.
			channel_shift_range=0.,  # Set range for random channel shifts.
			# Set mode for filling points outside the input boundaries.
			fill_mode='nearest',
			cval=0.,  # Value used for fill_mode = 'constant'.
			horizontal_flip=True,  # Randomly flip images.
			vertical_flip=False,  # Randomly flip images.
			# Set rescaling factor (applied before any other transformation).
			rescale=None,
			# Set function that will be applied on each input.
			preprocessing_function=None,
			# Image data format, either 'channels_first' or 'channels_last'.
			data_format=None,
			# Fraction of images reserved for validation (strictly between 0 and 1).
			validation_split=0.0)

		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)

		# Fit the model on the batches generated by datagen.flow().
		model.fit_generator(
			datagen.flow(x_train, y_train, batch_size=batch_size),
			epochs=epochs,
			validation_data=(x_test, y_test),
			use_multiprocessing=True,
			workers=4)

def main():
	cifar10_cnn_capsule_network_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
