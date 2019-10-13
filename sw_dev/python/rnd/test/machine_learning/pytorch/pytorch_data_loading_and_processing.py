#!/usr/bin/env python

from __future__ import print_function, division
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import skimage
import matplotlib.pyplot as plt
import PIL.Image

# Ignore warnings.
import warnings
warnings.filterwarnings('ignore')

plt.ion()  # Interactive mode.

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def mnist_dataset_test():
	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.ToTensor()
	#transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	#--------------------
	train_set = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(train_loader)
	images, labels = data_iter.next()
	images, labels = images.numpy(), labels.numpy()
	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	test_set = torchvision.datasets.MNIST(root='./mnist', train=False, download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(test_loader)
	images, labels = data_iter.next()
	images, labels = images.numpy(), labels.numpy()
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(test_loader):
	#	batch_inputs, batch_outputs = batch_data

# REF [site] >> https://pytorch.org/docs/stable/torchvision/datasets.html
def imagenet_dataset_test():
	if 'posix' == os.name:
		imagenet_dir_path = '/home/sangwook/my_dataset/pattern_recognition/imagenet'
	else:
		imagenet_dir_path = 'E:/dataset/pattern_recognition/imagenet'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.ToTensor()
	#transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

	#--------------------
	train_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='train', download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(train_loader)
	images, labels = data_iter.next()
	images, labels = images.numpy(), labels.numpy()
	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	test_set = torchvision.datasets.ImageNet(root=imagenet_dir_path, split='val', download=True, transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(test_loader)
	images, labels = data_iter.next()
	images, labels = images.numpy(), labels.numpy()
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(labels.shape, labels.dtype, np.min(labels), np.max(labels)))

	#for batch_step, batch_data in enumerate(test_loader):
	#	batch_inputs, batch_outputs = batch_data

def coco_dataset_captions_test():
	if 'posix' == os.name:
		coco_dir_path = '/home/sangwook/my_dataset/pattern_recognition/coco'
	else:
		coco_dir_path = 'E:/dataset/pattern_recognition/coco'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	train_set = torchvision.datasets.CocoCaptions(root=os.path.join(coco_dir_path, 'train2014'), annFile=os.path.join(coco_dir_path, 'annotations/captions_train2014.json'), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor, list of tuples.
	images = images.numpy()
	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: type = {}, length = {}, type = {}, length = {}.'.format(type(labels), len(labels), type(labels[0]), len(labels[0])))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	test_set = torchvision.datasets.CocoCaptions(root=os.path.join(coco_dir_path, 'val2014'), annFile=os.path.join(coco_dir_path, 'annotations/captions_val2014.json'), transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(test_loader)
	images, labels = data_iter.next()  # torch.Tensor, list of tuples.
	images = images.numpy()
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: type = {}, length = {}, type = {}, length = {}.'.format(type(labels), len(labels), type(labels[0]), len(labels[0])))

	#for batch_step, batch_data in enumerate(test_loader):
	#	batch_inputs, batch_outputs = batch_data

def coco_dataset_detection_test():
	if 'posix' == os.name:
		coco_dir_path = '/home/sangwook/my_dataset/pattern_recognition/coco'
	else:
		coco_dir_path = 'E:/dataset/pattern_recognition/coco'

	batch_size = 32
	shuffle = True
	num_workers = 4

	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR),
		torchvision.transforms.ToTensor(),
		#torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
	])

	#--------------------
	train_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_dir_path, 'train2014'), annFile=os.path.join(coco_dir_path, 'annotations/instances_train2014.json'), transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(train_loader)
	images, labels = data_iter.next()  # torch.Tensor, list of tuples.
	images = images.numpy()
	print('Train image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Train label: type = {}, length = {}, type = {}, length = {}.'.format(type(labels), len(labels), type(labels[0]), len(labels[0])))

	#for batch_step, batch_data in enumerate(train_loader):
	#	batch_inputs, batch_outputs = batch_data

	#--------------------
	test_set = torchvision.datasets.CocoDetection(root=os.path.join(coco_dir_path, 'val2014'), annFile=os.path.join(coco_dir_path, 'annotations/instances_val2014.json'), transform=transform)
	test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

	data_iter = iter(test_loader)
	images, labels = data_iter.next()  # torch.Tensor, list of tuples.
	images = images.numpy()
	print('Test image: shape = {}, dtype = {}, (min, max) = ({}, {}).'.format(images.shape, images.dtype, np.min(images), np.max(images)))
	print('Test label: type = {}, length = {}, type = {}, length = {}.'.format(type(labels), len(labels), type(labels[0]), len(labels[0])))

	#for batch_step, batch_data in enumerate(test_loader):
	#	batch_inputs, batch_outputs = batch_data

def show_landmarks(image, landmarks):
	"""Show image with landmarks"""
	plt.imshow(image)
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
	plt.pause(0.001)  # Pause a bit so that plots are updated.

# REF [site] >> https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
def simple_example():
	landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

	n = 65
	img_name = landmarks_frame.iloc[n, 0]
	landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
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

		image = image[top: top + new_h,
					  left: left + new_w]

		landmarks = landmarks - [left, top]

		return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		image, landmarks = sample['image'], sample['landmarks']

		# Wwap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		image = image.transpose((2, 0, 1))
		return {'image': torch.from_numpy(image),
				'landmarks': torch.from_numpy(landmarks)}

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
	transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
												root_dir='data/faces/',
												transform=torchvision.transforms.Compose([
													Rescale(256),
													RandomCrop(224),
													ToTensor()
												]))

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
			plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
						landmarks_batch[i, :, 1].numpy() + grid_border_size,
						s=10, marker='.', c='r')

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

def main():
	mnist_dataset_test()
	#imagenet_dataset_test()
	#coco_dataset_captions_test()
	#coco_dataset_detection_test()

	#simple_example()
	#dataset_example()
	#dataset_with_data_processing_example()

	#data_processing()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
