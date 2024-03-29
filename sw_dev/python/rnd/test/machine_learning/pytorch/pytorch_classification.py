#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)  # For 28x28 input.
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class MyNet(nn.Module):
	def __init__(self):
		super(MyNet, self).__init__()

		self.pool = nn.MaxPool2d(2, 2)
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.batchnorm21 = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Input: (batch size, channel, height, width).
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.batchnorm22 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Input: (batch size, channel, height, width).
		self.fc1 = nn.Linear(16 * 5 * 5, 120)  # For 28x28 input.
		self.batchnorm11 = nn.BatchNorm1d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Input: (batch size, feature dim) or (batch size, feature dim, time-steps).
		self.fc2 = nn.Linear(120, 84)
		self.batchnorm12 = nn.BatchNorm1d(84, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  # Input: (batch size, feature dim) or (batch size, feature dim, time-steps).
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		"""
		x = self.pool(F.dropout(F.relu(self.batchnorm21(self.conv1(x)))))
		x = self.pool(F.dropout(F.relu(self.batchnorm22(self.conv2(x)))))
		x = x.view(-1, 16 * 5 * 5)
		x = F.dropout(F.relu(self.batchnorm11(self.fc1(x))))
		x = F.dropout(F.relu(self.batchnorm12(self.fc2(x))))
		x = self.fc3(x)
		#x = F.log_softmax(self.fc3(x), dim=-1)
		return x
		"""
		x = self.pool(F.relu(self.batchnorm21(self.conv1(x))))
		x = self.pool(F.relu(self.batchnorm22(self.conv2(x))))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.batchnorm11(self.fc1(x)))
		x = F.relu(self.batchnorm12(self.fc2(x)))
		x = self.fc3(x)
		#x = F.log_softmax(self.fc3(x), dim=-1)
		return x

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def cifar10_on_cpu():
	batch_size, num_epochs = 4, 2

	# Load and normalize CIFAR10.
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a Convolutional Neural Network.

	#net = Net()
	net = MyNet()

	print('Model is on {}.'.format(next(net.parameters()).device))

	#--------------------
	# Define a Loss function and optimizer.

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	#--------------------
	# Train the network.

	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# Get the inputs; data is a list of [inputs, labels].
			inputs, labels = data

			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Forward + backward + optimize.
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Print statistics.
			running_loss += loss.item()
			if i % 2000 == 1999:  # Print every 2000 mini-batches.
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	#--------------------
	# Test the network on the test data.

	dataiter = iter(testloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	outputs = net(images)

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def cifar10_on_gpu():
	batch_size, num_epochs = 4, 2

	# Load and normalize CIFAR10.
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def imshow(img):
		img = img / 2 + 0.5  # Unnormalize.
		npimg = img.numpy()
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		plt.show()

	# Get some random training images.
	dataiter = iter(trainloader)
	images, labels = dataiter.next()

	# Show images.
	imshow(torchvision.utils.make_grid(images))
	# Print labels.
	print(' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	#--------------------
	# Define a Convolutional Neural Network.

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# Assuming that we are on a CUDA machine, this should print a CUDA device.
	print('Device: {}.'.format(device))

	net = Net()
	#net = MyNet()
	net.to(device)

	print('Model is on {}.'.format(next(net.parameters()).device))

	#--------------------
	# Define a Loss function and optimizer.

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

	#--------------------
	# Train the network.

	for epoch in range(num_epochs):  # Loop over the dataset multiple times.
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# Get the inputs; data is a list of [inputs, labels].
			inputs, labels = data[0].to(device), data[1].to(device)

			# Zero the parameter gradients.
			optimizer.zero_grad()

			# Forward + backward + optimize.
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Print statistics.
			running_loss += loss.item()
			if i % 2000 == 1999:  # Print every 2000 mini-batches.
				print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
				running_loss = 0.0

	print('Finished Training')

	#--------------------
	# Test the network on the test data.

	dataiter = iter(testloader)
	images, labels = dataiter.next()

	# Print images.
	imshow(torchvision.utils.make_grid(images))
	print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))

	# Now let us see what the neural network thinks these examples above are.
	outputs = net(images.to(device))

	_, predicted = torch.max(outputs, 1)
	print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(labels))))

	# Let us look at how the network performs on the whole dataset.
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

	# What are the classes that performed well, and the classes that did not perform well.
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			outputs = net(images)
			_, predicted = torch.max(outputs, 1)
			c = (predicted == labels).squeeze()
			for i in range(len(labels)):
				label = labels[i]
				class_correct[label] += c[i].item()
				class_total[label] += 1

	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def main():
	# Image classification.
	#	REF [site] >> https://github.com/pytorch/vision/tree/main/references/classification
	#
	#	AlexNet and VGG
	#	GoogLeNet
	#	Inception V3
	#	ResNet
	#	ResNeXt
	#	MobileNetV2
	#	MobileNetV3 Large & Small
	#	EfficientNet-V1
	#	EfficientNet-V2
	#	RegNet
	#	Vision Transformer
	#	ConvNeXt
	#	SwinTransformer
	#	SwinTransformer V2
	#	MaxViT
	#	ShuffleNet V2
	#
	#	Mixed precision training:
	#
	#	Quantized:
	#	Quantized ShuffleNet V2
	#	QAT MobileNetV2
	#	QAT MobileNetV3

	# Video classification.
	#	REF [site] >> https://github.com/pytorch/vision/tree/main/references/video_classification
	#
	#	Video ResNet
	#	S3D

	# REF [file] >> ./pytorch_model.py.

	#cifar10_on_cpu()
	cifar10_on_gpu()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
