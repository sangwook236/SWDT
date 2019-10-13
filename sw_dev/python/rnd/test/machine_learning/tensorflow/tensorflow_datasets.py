#!/usr/bin/env python

# REF [site] >>
#	https://www.tensorflow.org/datasets
#	https://www.tensorflow.org/datasets/overview
#	https://www.tensorflow.org/datasets/catalog/overview
#	https://github.com/tensorflow/datasets

import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_datasets as tfds

def mnist_dataset_test():
	# tfds works in both Eager and Graph modes.
	tf.enable_eager_execution()

	# See available datasets.
	print('Available datasets =', tfds.list_builders())

	#--------------------
	# Construct a tf.data.Dataset.
	if True:
		#datasets = tfds.load(name='mnist')  # A dictionary of tf.data.Dataset's.
		dataset = tfds.load(name='mnist', split=tfds.Split.TRAIN)  # tf.data.Dataset.
		#dataset = tfds.load(name='mnist', split=tfds.Split.TRAIN, batch_size=37, download=True)  # tf.data.Dataset.
		print('Dataset: type = {}.'.format(type(dataset)))
		print('Dataset = {}.'.format(dataset))

		for example in dataset:
			image, label = example['image'], example['label']

			print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
			print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))
			break

		# Build your input pipeline.
		dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)  # tf.data.Dataset.
		print('Dataset: type = {}.'.format(type(dataset)))
		print('Dataset = {}.'.format(dataset))

		for example in dataset.take(1):
			image, label = example['image'], example['label']

			print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
			print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))
	else:
		dataset = tfds.load(name='mnist', split=tfds.Split.TRAIN, as_supervised=True)  # tf.data.Dataset.
		#dataset = tfds.load(name='mnist', split=tfds.Split.TRAIN, batch_size=37, download=True, as_supervised=True)  # tf.data.Dataset.
		print('Dataset: type = {}.'.format(type(dataset)))
		print('Dataset = {}.'.format(dataset))

		for image, label in dataset:
			print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
			print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))
			break

		# Build your input pipeline.
		dataset = dataset.shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
		print('Dataset: type = {}.'.format(type(dataset)))
		print('Dataset = {}.'.format(dataset))

		for image, label in dataset.take(1):
			print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
			print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	#--------------------
	# Load a given dataset by name, along with the DatasetInfo.
	datasets, info = tfds.load('mnist', with_info=True)  # A dictionary of tf.data.Dataset's & tfds.core.DatasetInfo.

	print('Datasets: keys = {}.'.format(datasets.keys()))
	print('#classes =', info.features['label'].num_classes)
	print('#train examples =', info.splits['train'].num_examples)
	print('#test examples =', info.splits['test'].num_examples)

	train_data, test_data = datasets['train'], datasets['test']
	print('Train dataset: type = {}.'.format(type(train_data)))
	print('Test dataset: type = {}.'.format(type(test_data)))

	for example in train_data.take(1):
		image, label = example['image'], example['label']

		print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	for example in test_data.take(1):
		image, label = example['image'], example['label']

		print('\tTest image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTest label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	#--------------------
	# Access a builder directly.
	builder = tfds.builder('mnist')

	print('#classes =', builder.info.features['label'].num_classes)
	print('#train examples =', builder.info.splits['train'].num_examples)
	#print('#test examples =', builder.info.splits['test'].num_examples)

	builder.download_and_prepare()
	datasets = builder.as_dataset()  # A dictionary of tf.data.Dataset's.

	print('Datasets: keys = {}.'.format(datasets.keys()))
	print('Train dataset: type = {}.'.format(type(datasets['train'])))
	#print('Test dataset: type = {}.'.format(type(datasets['test'])))

	# NumPy arrays.
	np_datasets = tfds.as_numpy(datasets)  # A dictionary of np.arrays.

	print('Datasets: keys = {}.'.format(np_datasets.keys()))
	print('Train dataset: type = {}.'.format(type(np_datasets['train'])))
	#print('Test dataset: type = {}.'.format(type(np_datasets['test'])))

	for example in np_datasets['train']:
		image, label = example['image'], example['label']

		print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))
		break

def imagenet_dataset_test():
	if 'posix' == os.name:
		imagenet_dir_path = '/home/sangwook/my_dataset/pattern_recognition/imagenet'
	else:
		imagenet_dir_path = 'E:/dataset/pattern_recognition/imagenet'

	datasets, info = tfds.load('imagenet2012', data_dir=imagenet_dir_path, download=True, with_info=True)

def coco_dataset_test():
	if 'posix' == os.name:
		coco_dir_path = '/home/sangwook/my_dataset/pattern_recognition/coco'
	else:
		coco_dir_path = 'E:/dataset/pattern_recognition/coco'

	datasets, info = tfds.load('coco', data_dir=os.path.join(coco_dir_path, 'train2014'), download=False, with_info=True)

def main():
	#mnist_dataset_test()

	#imagenet_dataset_test()
	coco_dataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
