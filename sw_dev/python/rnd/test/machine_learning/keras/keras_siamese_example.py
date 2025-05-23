#!/usr/bin/env python
# coding: UTF-8

from __future__ import absolute_import
from __future__ import print_function

import random
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

def euclidean_distance(vects):
	x, y = vects
	sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
	return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
	shape1, shape2 = shapes
	return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
	'''Contrastive loss from Hadsell-et-al.'06
	http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	'''
	margin = 1
	square_pred = K.square(y_pred)
	margin_square = K.square(K.maximum(margin - y_pred, 0))
	return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(x, digit_indices, num_classes):
	'''Positive and negative pair creation.
	Alternates between positive and negative pairs.
	'''
	pairs = []
	labels = []
	n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
	for d in range(num_classes):
		for i in range(n):
			z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
			pairs += [[x[z1], x[z2]]]
			inc = random.randrange(1, num_classes)
			dn = (d + inc) % num_classes
			z1, z2 = digit_indices[d][i], digit_indices[dn][i]
			pairs += [[x[z1], x[z2]]]
			labels += [1, 0]
	return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
	'''Base network to be shared (eq. to feature extraction).
	'''
	input = Input(shape=input_shape)
	x = Flatten()(input)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu')(x)
	return Model(input, x)

def compute_accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	pred = y_pred.ravel() < 0.5
	return np.mean(pred == y_true)

def accuracy(y_true, y_pred):
	'''Compute classification accuracy with a fixed threshold on distances.
	'''
	return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

# REF [site] >> ${KERAS_HOME}/examples/mnist_siamese.py
# REF [paper] >> "Dimensionality Reduction by Learning an Invariant Mapping", CVPR 2006.
def siamese_mnist_example():
	num_classes = 10
	epochs = 20

	# The data, split between train and test sets.
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	input_shape = x_train.shape[1:]

	# Create training+test positive and negative pairs.
	digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
	tr_pairs, tr_y = create_pairs(x_train, digit_indices, num_classes)

	digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
	te_pairs, te_y = create_pairs(x_test, digit_indices, num_classes)

	# Network definition.
	base_network = create_base_network(input_shape)

	input_a = Input(shape=input_shape)
	input_b = Input(shape=input_shape)

	# Because we re-use the same instance 'base_network', the weights of the network will be shared across the two branches.
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)

	distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

	model = Model([input_a, input_b], distance)

	# Train.
	rms = RMSprop()
	model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
	model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
		batch_size=128,
		epochs=epochs,
		validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

	# Compute final accuracy on training and test sets.
	y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
	tr_acc = compute_accuracy(tr_y, y_pred)
	y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
	te_acc = compute_accuracy(te_y, y_pred)

	print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
	print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

def main():
	siamese_mnist_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
