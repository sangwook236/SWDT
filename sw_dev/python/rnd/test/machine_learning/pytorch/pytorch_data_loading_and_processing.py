#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function, division
import os, math, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import skimage
import PIL.Image
import matplotlib.pyplot as plt

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')

plt.ion()  # Interactive mode.

# REF [site] >> https://pytorch.org/docs/stable/data.html
# PyTorch supports two different types of datasets:
# Map-style datasets
#	A map-style dataset is one that implements the __getitem__() and __len__() protocols, and represents a map from (possibly non-integral) indices/keys to data samples.
#	For example, such a dataset, when accessed with dataset[idx], could read the idx-th image and its corresponding label from a folder on the disk.
#	torch.utils.data.Sampler classes are used to specify the sequence of indices/keys used in data loading. 
# Iterable-style datasets
#	An iterable-style dataset is an instance of a subclass of IterableDataset that implements the __iter__() protocol, and represents an iterable over data samples.
#	This type of datasets is particularly suitable for cases where random reads are expensive or even improbable, and where the batch size depends on the fetched data.
#	For example, such a dataset, when called iter(dataset), could return a stream of data reading from a database, a remote server, or even logs generated in real time.
#	When using an IterableDataset with multi-process data loading, the same dataset object is replicated on each worker process, and thus the replicas must be configured differently to avoid duplicated data.
#	For iterable-style datasets, data loading order is entirely controlled by the user-defined iterable. This allows easier implementations of chunk-reading and dynamic batch size (e.g., by yielding a batched sample at each time)

class ReturnNoneDataset(torch.utils.data.Dataset):
	def __init__(self, num_data, transform=None, target_transform=None):
		self.num_data = num_data
		self.transform = transform
		self.target_transform = target_transform

		assert self.num_data > 0

	def __len__(self):
		return self.num_data

	def __getitem__(self, idx):
		if idx % 2 == 0:
			return None
		x = [idx, 0, -idx]
		y = [idx**2]

		if self.transform:
			x = self.transform(x)
		if self.target_transform:
			y = self.target_transform(y)

		return x, y

def return_none_dataset_test():
	dataset = ReturnNoneDataset(num_data=20, transform=torch.IntTensor, target_transform=torch.IntTensor)
	print('#data = {}.'.format(len(dataset)))

	#for idx, dat in enumerate(dataset):  # Not correctly working.
	#	print('{} -> {}.'.format(idx, dat))
	for idx in range(len(dataset)):
		print('Datum {}: {}.'.format(idx, dataset[idx]))

	#-----
	def collate_except_none(batch):
		batch = list(filter(lambda x: x is not None, batch))
		return torch.utils.data.default_collate(batch) if batch else None
		#return torch.utils.data.default_collate([])  # IndexError: list index out of range.
		#return []  # OK.

	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, num_workers=4, collate_fn=collate_except_none)
	print('#batches = {}.'.format(len(dataloader)))

	for idx, batch in enumerate(dataloader):
		print('Batch {}: {}.'.format(idx, batch))

def dataset_test():
	# NOTE [info] >>
	#	Each worker process processes a subset of the data, which is of size num_data / (#worker processes * #gpus).
	#	Each worker process is assigned a subset of the indices of the data.

	print('The main process ID = {}.'.format(os.getpid()))

	class MyDataset1(torch.utils.data.Dataset):
		# NOTE [info] >> This method is called only once in the main process, not in the worker processes, when an instance is created.
		def __init__(self, data):
			super().__init__()

			self.data = data

			print('MyDataset1.__init__() is called in {}.'.format(os.getpid()))

		# NOTE [info] >> This method is called only in the main process, not in the worker processes.
		def __len__(self):
			print('MyDataset1.__len__() is called in {}.'.format(os.getpid()))
			return len(self.data)

		# NOTE [info] >> This method is called in the main process (single-process loading) or in every worker process (multi-process loading).
		def __getitem__(self, idx):
			print('MyDataset1.__getitem__(idx={}) is called in {}.'.format(idx, os.getpid()))
			return self.data[idx]

	dataset = MyDataset1(data=[3.0, 4.0, 5.0, 6.0])
	print('#data = {}.'.format(len(dataset)))
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)
	print('#steps = {}.'.format(len(dataloader)))

	# Single-process loading.
	print('-----1-1')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	print('-----1-1 (batch_size = 3)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=0)
	print(list(dataloader))  # [[3, 4, 5], [6]].
	print(list(dataloader))  # [[3, 4, 5], [6]].

	# Multi-process loading with two worker processes.
	print('-----1-2')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
	print(list(dataloader))  # [3, 5, 4, 6].
	print(list(dataloader))  # [3, 5, 4, 6].

	# Multi-process loading with two worker processes.
	print('-----1-2 (batch_size = 3)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=2)
	print(list(dataloader))  # [[3, 4, 5], [6]].
	print(list(dataloader))  # [[3, 4, 5], [6]].

	# With even more workers.
	print('-----1-3')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=20)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	# With even more workers.
	print('-----1-3 (batch_size = 3)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=20)
	print(list(dataloader))  # [[3, 4, 5], [6]].
	print(list(dataloader))  # [[3, 4, 5], [6]].

	#--------------------
	# Multiple outputs.
	class MyDataset2(torch.utils.data.Dataset):
		def __init__(self, data1, data2):
			super().__init__()

			assert len(data1) == len(data2), 'data1 and data2 must have the same length, {} != {}'.format(len(data1), len(data2))
			self.data1 = data1
			self.data2 = data2
			#self.return_items = ['data1', 'data2']

		def __len__(self):
			return len(self.data1)

		def __getitem__(self, idx):
			#return self.data1[idx], self.data2[idx]
			#return idx, self.data1[idx], self.data2[idx]
			return {'data1': self.data1[idx], 'data2': self.data2[idx]}
			#return {'idx': idx, 'data1': self.data1[idx], 'data2': self.data2[idx]}

	dataset = MyDataset2(data1=[3.0, 4.0, 5.0, 6.0], data2=[-3.0, -4.0, -5.0, -6.0])
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)
	print('#steps = {}.'.format(len(dataloader)))

	print('--------------------')
	print('-----2-1')
	#dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
	#dataloader = torch.utils.data.DataLoader(dataset, num_workers=20)
	print(list(dataloader))  # [[3, -3] [4, -4], [5, -5], [6, -6]] or [{'data1': 3, 'data2': -3}, {'data1': 4, 'data2': -4}, {'data1': 5, 'data2': -5}, {'data1': 6, 'data2': -6}].
	print(list(dataloader))  # [[3, -3] [4, -4], [5, -5], [6, -6]] or [{'data1': 3, 'data2': -3}, {'data1': 4, 'data2': -4}, {'data1': 5, 'data2': -5}, {'data1': 6, 'data2': -6}].

	print('-----2-1 (batch_size = 3)')
	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=0)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=2)
	#dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, num_workers=20)
	print(list(dataloader))  # [[[3, 4, 5], [-3, -4, -5]], [[6], [-6]]] or [{'data1': [3, 4, 5], 'data2': [-3, -4, -5]}, {'data1': [6], 'data2': [-6]}].
	print(list(dataloader))  # [[[3, 4, 5], [-3, -4, -5]], [[6], [-6]]] or [{'data1': [3, 4, 5], 'data2': [-3, -4, -5]}, {'data1': [6], 'data2': [-6]}].

# REF [site] >> https://pytorch.org/docs/stable/data.html
def iterable_dataset_test():
	print('The main process ID = {}.'.format(os.getpid()))

	# Split workload across all workers in __iter__().
	class MyIterableDataset1(torch.utils.data.IterableDataset):
		# NOTE [info] >> This method is called only once in the main process, not in the worker processes, when an instance is created.
		def __init__(self, start, end):
			super().__init__()

			assert end > start, 'This example code only works with end > start'
			self.start = start
			self.end = end

			print('MyIterableDataset1.__init__() is called in {}: work info = {}.'.format(os.getpid(), torch.utils.data.get_worker_info()))

		# NOTE [info] >> This method is called once in the main process (single-process loading) or in every worker process (multi-process loading) when starting iterating.
		def __iter__(self):
			worker_info = torch.utils.data.get_worker_info()

			if worker_info is None:  # Single-process data loading, return the full iterator.
				print('MyIterableDataset1.__iter__() is called in {}: worker_info = {}.'.format(os.getpid(), worker_info))
			else:
				print('MyIterableDataset1.__iter__() is called in {}: worker_info = {}.'.format(os.getpid(), worker_info))

			if worker_info is None:  # Single-process data loading, return the full iterator.
				iter_start = self.start
				iter_end = self.end
			else:  # In a worker process.
				# Split workload.
				per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
				worker_id = worker_info.id
				iter_start = self.start + worker_id * per_worker
				iter_end = min(iter_start + per_worker, self.end)
			return iter(range(iter_start, iter_end))

	# Should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
	dataset = MyIterableDataset1(start=3, end=7)

	#num_examples = len(dataset)  # NOTE [error] >> TypeError: object of type 'MyIterableDataset1' has no len().
	#dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, num_workers=0)  # NOTE [info] >> ValueError: DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True.
	#num_steps = len(dataloader)  # NOTE [error] >> TypeError: object of type 'MyIterableDataset1' has no len().

	# Single-process loading.
	print('-----1-1')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	# Multi-process loading with two worker processes.
	# Worker 0 fetched [3, 4]. Worker 1 fetched [5, 6].
	print('-----1-2')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
	print(list(dataloader))  # [3, 5, 4, 6].
	print(list(dataloader))  # [3, 5, 4, 6].

	# With even more workers.
	print('-----1-3')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=20)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	#--------------------
	# Split workload across all workers using worker_init_fn().
	class MyIterableDataset2(torch.utils.data.IterableDataset):
		def __init__(self, start, end):
			super().__init__()

			assert end > start, 'This example code only works with end > start'
			self.start = start
			self.end = end

			print('MyIterableDataset2.__init__() is called in {}: work info = {}.'.format(os.getpid(), torch.utils.data.get_worker_info()))

		def __iter__(self):
			print('MyIterableDataset2.__iter__() is called in {}: work info = {}.'.format(os.getpid(), torch.utils.data.get_worker_info()))
			return iter(range(self.start, self.end))

	print('--------------------')
	# Should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
	dataset = MyIterableDataset2(start=3, end=7)

	# Single-process loading.
	print('-----2-1')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	# Directly doing multi-process loading yields duplicate data.
	print('-----2-2')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
	print(list(dataloader))  # [3, 3, 4, 4, 5, 5, 6, 6].
	print(list(dataloader))  # [3, 3, 4, 4, 5, 5, 6, 6].

	# Define a 'worker_init_fn' that configures each dataset copy differently.
	# NOTE [info] >> This method is called once before calling IterableDataset.__iter__() in every worker process, not in the main process, when starting iterating.
	def worker_init_fn(worker_id):
		worker_info = torch.utils.data.get_worker_info()
		assert worker_id == worker_info.id

		print('worker_init_fn() is called in {}: work info = {}.'.format(os.getpid(), worker_info))

		dataset = worker_info.dataset  # The dataset copy in this worker process.
		overall_start = dataset.start
		overall_end = dataset.end
		# Configure the dataset to only process the split workload.
		per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
		dataset.start = overall_start + worker_id * per_worker
		dataset.end = min(dataset.start + per_worker, overall_end)

	# Single-process loading.
	print('-----2-3')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	# Multi-process loading with the custom 'worker_init_fn'.
	# Worker 0 fetched [3, 4]. Worker 1 fetched [5, 6].
	print('-----2-4')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, worker_init_fn=worker_init_fn)
	print(list(dataloader))  # [3, 5, 4, 6].
	print(list(dataloader))  # [3, 5, 4, 6].

	# With even more workers.
	print('-----2-5')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=20, worker_init_fn=worker_init_fn)
	print(list(dataloader))  # [3, 4, 5, 6].
	print(list(dataloader))  # [3, 4, 5, 6].

	#--------------------
	# Multiple outputs.
	class MyIterableDataset3(torch.utils.data.IterableDataset):
		def __init__(self, val):
			super().__init__()
			self.val = val

		def __iter__(self):
			data1, data2 = list(range(1, self.val + 1)), list(range(-1, -(self.val + 1), -1))
			assert len(data1) == len(data2)
			num_examples = len(data1)

			worker_info = torch.utils.data.get_worker_info()

			if worker_info is None:  # Single-process data loading, return the full iterator.
				return iter(zip(data1, data2))
			else:  # In a worker process.
				# Split workload.
				worker_id = worker_info.id
				num_examples_per_worker = math.ceil(num_examples / float(worker_info.num_workers))
				iter_start = worker_id * num_examples_per_worker
				iter_end = min(iter_start + num_examples_per_worker, num_examples)
				return iter(zip(data1[iter_start:iter_end], data2[iter_start:iter_end]))

	print('--------------------')
	dataset = MyIterableDataset3(val=5)

	# Single-process loading.
	print('-----3-1')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].

	print('-----3-1 (batch_size = 2)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
	print(list(dataloader))  # [[[1, 2], [-1, -2]], [[3, 4], [-3, -4]], [[5], [-5]]].
	print(list(dataloader))  # [[[1, 2], [-1, -2]], [[3, 4], [-3, -4]], [[5], [-5]]].

	# Multi-process loading with two worker processes.
	# Worker 0 fetched [3, 4]. Worker 1 fetched [5, 6].
	print('-----3-2')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=2)
	print(list(dataloader))  # [[1, -1], [4, -4], [2, -2], [5, -5], [3, -3]].
	print(list(dataloader))  # [[1, -1], [4, -4], [2, -2], [5, -5], [3, -3]].

	print('-----3-2 (batch_size = 2)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
	print(list(dataloader))  # [[[1, 2], [-1, -2]], [[4, 5], [-4, -5]], [[3], [-3]]].
	print(list(dataloader))  # [[[1, 2], [-1, -2]], [[4, 5], [-4, -5]], [[3], [-3]]].

	# With even more workers.
	print('-----3-3')
	dataloader = torch.utils.data.DataLoader(dataset, num_workers=20)
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].

	print('-----3-3 (batch_size = 2)')
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=20)
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].
	print(list(dataloader))  # [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5]].

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def mnist_dataset_test():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	mnist_dir_path = data_dir_path + '/language_processing/mnist'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.ToTensor()
	#transform = torchvision.transforms.Compose([
	#	torchvision.transforms.ToTensor(),
	#	torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	#])

	#--------------------
	print('Start creating MNIST dataset and data loader for train...')
	start_time = time.time()
	train_set = torchvision.datasets.MNIST(root=mnist_dir_path, train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating MNIST dataset and data loader for train: {} secs.'.format(time.time() - start_time))

	print('#train steps per epoch = {}.'.format(len(train_loader)))

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	print('Start creating MNIST dataset and data loader for test...')
	start_time = time.time()
	test_set = torchvision.datasets.MNIST(root=mnist_dir_path, train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating MNIST dataset and data loader for test: {} secs.'.format(time.time() - start_time))

	print('#test steps per epoch = {}.'.format(len(test_loader)))

	data_iter = iter(test_loader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Test image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(test_loader):
	#	batch_inputs, batch_outputs = batch_data

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def imagenet_dataset_test():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	imagenet_dir_path = data_dir_path + '/pattern_recognition/imagenet'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating ImageNet dataset and data loader for train...')
	start_time = time.time()
	train_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='train', download=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating ImageNet dataset and data loader for train: {} secs.'.format(time.time() - start_time))

	print('#train steps per epoch = {}.'.format(len(train_loader)))

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	print('Start creating ImageNet dataset and data loader for validation...')
	start_time = time.time()
	val_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='val', download=False, transform=transform)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating ImageNet dataset and data loader for validation: {} secs.'.format(time.time() - start_time))

	print('#validation steps per epoch = {}.'.format(len(val_loader)))

	data_iter = iter(val_loader)
	images, labels = data_iter.next()  # torch.Tensor & torch.Tensor.
	images, labels = images.numpy(), labels.numpy()
	print('Validation image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Validation label: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(val_loader):
	#	batch_inputs, batch_outputs = batch_data

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def coco_dataset_captions_test():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	coco_dir_path = data_dir_path + '/pattern_recognition/coco'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating COCO captions dataset and data loader for train...')
	start_time = time.time()
	train_set = torchvision.datasets.CocoCaptions(root=os.path.join(coco_dir_path, 'train2014'), annFile=os.path.join(coco_dir_path, 'annotations/captions_train2014.json'), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating COCO captions dataset and data loader for train: {} secs.'.format(time.time() - start_time))

	print('#train steps per epoch = {}.'.format(len(train_loader)))

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor & a list of 5 tuples, each of which has strings of batch size.
	images = images.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Type = {}, length = {}.'.format(type(labels), len(labels)))
	#print('\tlen(labels[0]) = {}.'.format(len(labels[0])))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	print('Start creating COCO captions dataset and data loader for validation...')
	start_time = time.time()
	val_set = torchvision.datasets.CocoCaptions(root=os.path.join(coco_dir_path, 'val2014'), annFile=os.path.join(coco_dir_path, 'annotations/captions_val2014.json'), transform=transform)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating COCO captions dataset and data loader for validation: {} secs.'.format(time.time() - start_time))

	print('#validation steps per epoch = {}.'.format(len(val_loader)))

	data_iter = iter(val_loader)
	images, labels = data_iter.next()  # torch.Tensor & a list of 5 tuples, each of which has strings of batch size.
	images = images.numpy()
	print('Validation image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Validation label: Type = {}, length = {}.'.format(type(labels), len(labels)))
	#print('\tlen(labels[0]) = {}.'.format(len(labels[0])))

	#for batch_step, batch_data in enumerate(val_loader):
	#	batch_inputs, batch_outputs = batch_data

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def coco_dataset_detection_test():
	if 'posix' == os.name:
		data_dir_path = '/home/sangwook/my_dataset'
	else:
		data_dir_path = 'E:/dataset'
	coco_dir_path = data_dir_path + '/pattern_recognition/coco'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	print('Start creating COCO detection dataset and data loader for train...')
	start_time = time.time()
	train_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_dir_path, 'train2014'), annFile=os.path.join(coco_dir_path, 'annotations/instances_train2014.json'), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating COCO detection dataset and data loader for train: {} secs.'.format(time.time() - start_time))

	print('#train steps per epoch = {}.'.format(len(train_loader)))

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor & a list of dicts of #detections, each of which has 7 elements ('segmentation' (1 * ? * batch size), 'area' (batch size), 'iscrowd' (batch size), 'image_id' (batch size), 'bbox' (4 * batch size), 'category_id' (batch size), and 'id' (batch size)).
	images = images.numpy()
	print('Train image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: Type = {}, length = {}.'.format(type(labels), len(labels)))
	#print('Train label: Keys = {}.'.format(labels[0].keys()))
	for idx, label in enumerate(labels):
		print('\tDetection {}: {}, {}, {}, {}, {}, {}, {}.'.format(idx, len(label['segmentation']), len(label['area']), len(label['iscrowd']), len(label['image_id']), len(label['bbox']), len(label['category_id']), len(label['id'])))
		#print('\t\tLengths: {}, {}, {}.'.format(len(label['segmentation'][0]), len(label['segmentation'][0][1]), len(label['bbox'][0])))
		bboxes = list()
		for bbox in label['bbox']:
			bboxes.append(bbox.numpy())
		print('\t\tBounding boxes = {}.'.format(np.vstack(bboxes).shape))
		segs = list()
		for seg in label['segmentation'][0]:
			segs.append(seg.numpy())
		print('\t\tSegmentations = {}.'.format(np.vstack(segs).shape))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	print('Start creating COCO detection dataset and data loader for validation...')
	start_time = time.time()
	val_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_dir_path, 'val2014'), annFile=os.path.join(coco_dir_path, 'annotations/instances_val2014.json'), transform=transform)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
	print('End creating COCO detection dataset and data loader for validation: {} secs.'.format(time.time() - start_time))

	print('#validation steps per epoch = {}.'.format(len(val_loader)))

	data_iter = iter(val_loader)
	images, labels = data_iter.next()  # torch.Tensor & a list of dicts of #detections, each of which has 7 elements ('segmentation' (1 * ? * batch size), 'area' (batch size), 'iscrowd' (batch size), 'image_id' (batch size), 'bbox' (4 * batch size), 'category_id' (batch size), and 'id' (batch size)).
	images = images.numpy()
	print('Validation image: Shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Validation label: Type = {}, length = {}.'.format(type(labels), len(labels)))
	#print('Validation label: Keys = {}.'.format(labels[0].keys()))
	for idx, label in enumerate(labels):
		print('\tDetection {}: {}, {}, {}, {}, {}, {}, {}.'.format(idx, len(label['segmentation']), len(label['area']), len(label['iscrowd']), len(label['image_id']), len(label['bbox']), len(label['category_id']), len(label['id'])))
		#print('\t\tLengths: {}, {}, {}.'.format(len(label['segmentation'][0]), len(label['segmentation'][0][1]), len(label['bbox'][0])))
		bboxes = list()
		for bbox in label['bbox']:
			bboxes.append(bbox.numpy())
		print('\t\tBounding Boxes = {}.'.format(np.vstack(bboxes).shape))
		segs = list()
		for seg in label['segmentation'][0]:
			segs.append(seg.numpy())
		print('\t\tSegmentations = {}.'.format(np.vstack(segs).shape))

	#for batch_step, batch_data in enumerate(val_loader):
	#	batch_inputs, batch_outputs = batch_data

def show_landmarks(image, landmarks):
	"""Show image with landmarks"""
	plt.imshow(image)
	plt.scatter(landmarks[:,0], landmarks[:,1], s=10, marker='.', c='r')
	plt.pause(0.001)  # Pause a bit so that plots are updated.

# REF [site] >> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def simple_example():
	landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

	n = 65
	img_name = landmarks_frame.iloc[n,0]
	landmarks = landmarks_frame.iloc[n,1:].as_matrix()
	landmarks = landmarks.astype('float').reshape(-1, 2)

	print('Image name: {}'.format(img_name))
	print('Landmarks shape: {}'.format(landmarks.shape))
	print('First 4 Landmarks: {}'.format(landmarks[:4]))

	plt.figure()
	show_landmarks(skimage.io.imread(os.path.join('data/faces/', img_name)), landmarks)
	plt.show()

class FaceLandmarksDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, csv_file, root_dir, transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.landmarks_frame = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.landmarks_frame)

	def __getitem__(self, idx):
		img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
		image = skimage.io.imread(img_name)
		landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
		landmarks = landmarks.astype('float').reshape(-1, 2)
		sample = {'image': image, 'landmarks': landmarks}

		if self.transform:
			sample = self.transform(sample)

		return sample

# REF [site] >> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def dataset_example():
	face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

	fig = plt.figure()

	for i in range(len(face_dataset)):
		sample = face_dataset[i]

		print(i, sample['image'].shape, sample['landmarks'].shape)

		ax = plt.subplot(1, 4, i + 1)
		plt.tight_layout()
		ax.set_title('Sample #{}'.format(i))
		ax.axis('off')
		show_landmarks(**sample)

		if 3 == i:
			plt.show()
			break

class Rescale(object):
	"""Rescale the image in a sample to a given size.

	Args:
		output_size (tuple or int): Desired output size. If tuple, output is
			matched to output_size. If int, smaller of image edges is matched
			to output_size keeping aspect ratio the same.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = skimage.transform.resize(image, (new_h, new_w))

		# h and w are swapped for landmarks because for images,
		# x and y axes are axis 1 and 0 respectively.
		landmarks = landmarks * [new_w / w, new_h / h]

		return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
	"""Crop randomly the image in a sample.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop is made.
	"""

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]

		landmarks = landmarks - [left, top]

		return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		# Swap channel axis:
		#	NumPy image: H x W x C.
		#	Torch image: C x H x W.
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

# REF [site] >> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def dataset_with_data_processing_example():
	# Compose transforms.
	scale = Rescale(256)
	crop = RandomCrop(128)
	composed = torchvision.transforms.Compose([Rescale(256), RandomCrop(224)])

	# Apply each of the above transforms on sample.
	face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

	fig = plt.figure()
	sample = face_dataset[65]
	for i, tsfrm in enumerate([scale, crop, composed]):
		transformed_sample = tsfrm(sample)

		ax = plt.subplot(1, 3, i + 1)
		plt.tight_layout()
		ax.set_title(type(tsfrm).__name__)
		show_landmarks(**transformed_sample)

	plt.show()

	# Iterate through the dataset.
	transformed_dataset = FaceLandmarksDataset(
		csv_file='data/faces/face_landmarks.csv',
		root_dir='data/faces/',
		transform=torchvision.transforms.Compose([
			Rescale(256),
			RandomCrop(224),
			ToTensor()
		])
	)

	for i in range(len(transformed_dataset)):
		sample = transformed_dataset[i]

		print(i, sample['image'].size(), sample['landmarks'].size())

		if 3 == i:
			break

	#--------------------
	dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

	# Helper function to show a batch.
	def show_landmarks_batch(sample_batched):
		"""Show image with landmarks for a batch of samples."""
		images_batch, landmarks_batch =  sample_batched['image'], sample_batched['landmarks']
		batch_size = len(images_batch)
		im_size = images_batch.size(2)
		grid_border_size = 2

		grid = torchvision.utils.make_grid(images_batch)
		plt.imshow(grid.numpy().transpose((1, 2, 0)))

		for i in range(batch_size):
			plt.scatter(
				landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
				landmarks_batch[i, :, 1].numpy() + grid_border_size,
				s=10, marker='.', c='r'
			)

			plt.title('Batch from dataloader')

	for i_batch, sample_batched in enumerate(dataloader):
		print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

		# Observe 4th batch and stop.
		if 3 == i_batch:
			plt.figure()
			show_landmarks_batch(sample_batched)
			plt.axis('off')
			plt.ioff()
			plt.show()
			break

# REF [site] >> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def data_processing():
	data_transform = torchvision.transforms.Compose([
		torchvision.transforms.RandomSizedCrop(224),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	hymenoptera_dataset = torchvision.datasets.ImageFolder(root='data/hymenoptera_data/train', transform=data_transform)
	dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset, batch_size=4, shuffle=True, num_workers=4)

# REF [site] >>
#	https://pytorch.org/vision/main/transforms.html
#	https://pytorch.org/vision/main/auto_examples/plot_transforms.html
def transform_test():
	def plot(img, imgs, with_orig=True, title=None, row_title=None, **imshow_kwargs):
		if not isinstance(imgs[0], list):
			# Make a 2d grid even if there's just 1 row.
			imgs = [imgs]

		num_rows = len(imgs)
		num_cols = len(imgs[0]) + with_orig
		fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
		for row_idx, row in enumerate(imgs):
			row = [img] + row if with_orig else row
			for col_idx, img in enumerate(row):
				ax = axs[row_idx, col_idx]
				ax.imshow(np.asarray(img), **imshow_kwargs)
				ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

		if with_orig:
			axs[0, 0].set(title='Original image')
			axs[0, 0].title.set_size(8)
		if row_title is not None:
			for row_idx in range(num_rows):
				axs[row_idx, 0].set(ylabel=row_title[row_idx])

		if title is not None:
			plt.suptitle(title)
		plt.tight_layout()

	img = PIL.Image.open('./astronaut.jpg')

	padded_imgs = [torchvision.transforms.Pad(padding=padding)(img) for padding in (3, 10, 30, 50)]
	plot(img, padded_imgs, title='Pad')

	resized_imgs = [torchvision.transforms.Resize(size=size)(img) for size in (30, 50, 100, img.size)]
	plot(img, resized_imgs, title='Resize')

	center_crops = [torchvision.transforms.CenterCrop(size=size)(img) for size in (30, 50, 100, img.size)]
	plot(img, center_crops, title='CenterCrop')

	(top_left, top_right, bottom_left, bottom_right, center) = torchvision.transforms.FiveCrop(size=(100, 100))(img)
	plot(img, [top_left, top_right, bottom_left, bottom_right, center], title='FiveCrop')

	gray_img = torchvision.transforms.Grayscale()(img)
	plot(img, [gray_img], title='Grayscale', cmap='gray')

	jitter = torchvision.transforms.ColorJitter(brightness=.5, hue=.3)
	jitted_imgs = [jitter(img) for _ in range(4)]
	plot(img, jitted_imgs, title='ColorJitter')

	blurrer = torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
	blurred_imgs = [blurrer(img) for _ in range(4)]
	plot(img, blurred_imgs, title='GaussianBlur')

	perspective_transformer = torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
	perspective_imgs = [perspective_transformer(img) for _ in range(4)]
	plot(img, perspective_imgs, title='RandomPerspective')

	rotater = torchvision.transforms.RandomRotation(degrees=(0, 180))
	rotated_imgs = [rotater(img) for _ in range(4)]
	plot(img, rotated_imgs, title='RandomRotation')

	affine_transfomer = torchvision.transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
	affine_imgs = [affine_transfomer(img) for _ in range(4)]
	plot(img, affine_imgs, title='RandomAffine')

	'''
	elastic_transformer = torchvision.transforms.ElasticTransform(alpha=250.0)
	transformed_imgs = [elastic_transformer(img) for _ in range(2)]
	plot(img, transformed_imgs, title='Pad')
	'''

	cropper = torchvision.transforms.RandomCrop(size=(128, 128))
	crops = [cropper(img) for _ in range(4)]
	plot(img, crops, title='RandomCrop')

	resize_cropper = torchvision.transforms.RandomResizedCrop(size=(32, 32))
	resized_crops = [resize_cropper(img) for _ in range(4)]
	plot(img, resized_crops, title='RandomResizedCrop')

	inverter = torchvision.transforms.RandomInvert()
	invertered_imgs = [inverter(img) for _ in range(4)]
	plot(img, invertered_imgs, title='RandomInvert')

	posterizer = torchvision.transforms.RandomPosterize(bits=2)
	posterized_imgs = [posterizer(img) for _ in range(4)]
	plot(img, posterized_imgs, title='RandomPosterize')

	solarizer = torchvision.transforms.RandomSolarize(threshold=192.0)
	solarized_imgs = [solarizer(img) for _ in range(4)]
	plot(img, solarized_imgs, title='RandomSolarize')

	sharpness_adjuster = torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2)
	sharpened_imgs = [sharpness_adjuster(img) for _ in range(4)]
	plot(img, sharpened_imgs, title='RandomAdjustSharpness')

	autocontraster = torchvision.transforms.RandomAutocontrast()
	autocontrasted_imgs = [autocontraster(img) for _ in range(4)]
	plot(img, autocontrasted_imgs, title='RandomAutocontrast')

	equalizer = torchvision.transforms.RandomEqualize()
	equalized_imgs = [equalizer(img) for _ in range(4)]
	plot(img, equalized_imgs, title='RandomEqualize')

	#--------------------
	#augmenter = torchvision.transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=None)
	policies = [torchvision.transforms.AutoAugmentPolicy.CIFAR10, torchvision.transforms.AutoAugmentPolicy.IMAGENET, torchvision.transforms.AutoAugmentPolicy.SVHN]
	augmenters = [torchvision.transforms.AutoAugment(policy) for policy in policies]
	imgs = [[augmenter(img) for _ in range(4)] for augmenter in augmenters]
	row_title = [str(policy).split('.')[-1] for policy in policies]
	plot(img, imgs, title='AutoAugment', row_title=row_title)

	#augmenter = torchvision.transforms.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=31, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=None)
	augmenter = torchvision.transforms.RandAugment()
	imgs = [augmenter(img) for _ in range(4)]
	plot(img, imgs, title='RandAugment')

	#augmenter = torchvision.transforms.TrivialAugmentWide(num_magnitude_bins=31, interpolation=torchvision.transforms.InterpolationMode.NEAREST, fill=None)
	augmenter = torchvision.transforms.TrivialAugmentWide()
	imgs = [augmenter(img) for _ in range(4)]
	plot(img, imgs, title='TrivialAugmentWide')

	'''
	#augmenter = torchvision.transforms.AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True, interpolation=InterpolationMode.BILINEAR, fill=None)
	augmenter = torchvision.transforms.AugMix()
	imgs = [augmenter(img) for _ in range(4)]
	plot(img, imgs, title='AugMix')
	'''

	plt.show()

def main():
	#return_none_dataset_test()

	#dataset_test()
	#iterable_dataset_test()

	#--------------------
	#mnist_dataset_test()
	#imagenet_dataset_test()
	#coco_dataset_captions_test()
	#coco_dataset_detection_test()

	#simple_example()
	#dataset_example()
	#dataset_with_data_processing_example()

	#--------------------
	#data_processing()

	transform_test()  # Transforms & automatic augmentation transforms.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
