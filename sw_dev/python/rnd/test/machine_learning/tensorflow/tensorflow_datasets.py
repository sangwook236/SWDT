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
	print('Available datasets = {}.'.format(tfds.list_builders()))

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
	print('Splits: keys = {}.'.format(info.splits.keys()))
	print('Features: keys = {}.'.format(info.features.keys()))

	print('#classes = {}.'.format(info.features['label'].num_classes))
	print('#train examples = {}.'.format(info.splits['train'].num_examples))
	print('#test examples = {}.'.format(info.splits['test'].num_examples))

	train_dataset, test_dataset = datasets['train'], datasets['test']
	print('Train dataset: type = {}.'.format(type(train_dataset)))
	print('Test dataset: type = {}.'.format(type(test_dataset)))

	for example in train_dataset.take(1):
		image, label = example['image'], example['label']

		print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	for example in test_dataset.take(1):
		image, label = example['image'], example['label']

		print('\tTest image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTest label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	#--------------------
	# Access a builder directly.
	builder = tfds.builder('mnist')

	print('#classes = {}.'.format(builder.info.features['label'].num_classes))
	print('#train examples = {}.'.format(builder.info.splits['train'].num_examples))
	#print('#test examples = {}.'.format(builder.info.splits['test'].num_examples))

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
	# tfds works in both Eager and Graph modes.
	#tf.enable_eager_execution()

	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	imagenet_dir_path = data_dir_path + '/pattern_recognition/imagenet'

	datasets, info = tfds.load('imagenet2012', data_dir=imagenet_dir_path, download=False, with_info=True)

	print('Datasets: keys = {}.'.format(datasets.keys()))
	print('Splits: keys = {}.'.format(info.splits.keys()))
	print('Features: keys = {}.'.format(info.features.keys()))

	print('#classes = {}.'.format(info.features['label'].num_classes))
	print('#train examples = {}.'.format(info.splits['train'].num_examples))
	print('#test examples = {}.'.format(info.splits['test'].num_examples))

	train_dataset = datasets['train']
	print('Train dataset: type = {}.'.format(type(train_dataset)))
	for example in train_dataset.take(1):
		image, label = example['image'], example['label']

		print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTrain label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

	test_dataset = datasets['test']
	print('Test dataset: type = {}.'.format(type(test_dataset)))
	for example in test_dataset.take(1):
		image, label = example['image'], example['label']

		print('\tTest image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTest label: shape = {}, dtype = {}.'.format(label.shape, label.dtype))

def coco_dataset_test():
	# tfds works in both Eager and Graph modes.
	#tf.enable_eager_execution()

	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	coco_dir_path = data_dir_path + '/pattern_recognition/coco'

	datasets, info = tfds.load('coco', data_dir=coco_dir_path, download=False, with_info=True)

	print('Datasets: keys = {}.'.format(datasets.keys()))
	print('Splits: keys = {}.'.format(info.splits.keys()))
	print('Features: keys = {}.'.format(info.features.keys()))

	"""
	print('#classes = {}.'.format(info.features['image']))
	print('#classes = {}.'.format(info.features['image/id']))
	print('#classes = {}.'.format(info.features['image/filename']))
	print('#classes = {}.'.format(info.features['objects']))
	"""
	print('#train examples = {}.'.format(info.splits['train'].num_examples))
	print('#test examples = {}.'.format(info.splits['test'].num_examples))
	print('#test2015 examples = {}.'.format(info.splits['test2015'].num_examples))
	print('#validation examples = {}.'.format(info.splits['validation'].num_examples))

	train_dataset = datasets['train']
	print('Train dataset: type = {}.'.format(type(train_dataset)))
	#print('Train dataset: output shapes = {}.'.format(tf.data.get_output_shapes(train_dataset)))
	#print('Train dataset: output types = {}.'.format(tf.data.get_output_types(train_dataset)))
	#print('Train dataset: output classes = {}.'.format(tf.data.get_output_classes(train_dataset)))
	for example in train_dataset.take(1):
		image, image_id, image_filename, objects = example['image'], example['image/id'], example['image/filename'], example['objects']
		area, bbox, is_crowd, label = objects['area'], objects['bbox'], objects['is_crowd'], objects['label']

		print('\tTrain image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTrain bboxes: shape = {}, dtype = {}.'.format(bboxes.shape, bboxes.dtype))
		print('\tTrain labels: shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
		break

	test_dataset = datasets['test']
	print('Test dataset: type = {}.'.format(type(test_dataset)))
	for example in test_dataset.take(1):
		image, label = example['image'], example['image/id']

		print('\tTest image: shape = {}, dtype = {}.'.format(image.shape, image.dtype))
		print('\tTest bboxes: shape = {}, dtype = {}.'.format(bboxes.shape, bboxes.dtype))
		print('\tTest labels: shape = {}, dtype = {}.'.format(labels.shape, labels.dtype))
		break

def main():
	#mnist_dataset_test()

	#imagenet_dataset_test()
	coco_dataset_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
