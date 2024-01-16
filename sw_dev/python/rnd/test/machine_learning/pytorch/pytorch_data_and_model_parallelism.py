#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

class RandomDataset(Dataset):
	def __init__(self, size, length):
		self.len = length
		self.data = torch.randn(length, size)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return self.len

class Model(nn.Module):
	def __init__(self, input_size, output_size):
		super(Model, self).__init__()
		self.fc = nn.Linear(input_size, output_size)

	def forward(self, input):
		output = self.fc(input)
		print('\tIn Model: input size', input.size(), 'output size', output.size())
		return output

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
def data_parallel_tutorial():
	# It's natural to execute your forward, backward propagations on multiple GPUs.
	# However, Pytorch will only use one GPU by default.
	# You can easily run your operations on multiple GPUs by making your model run parallelly using DataParallel.
	#	model = nn.DataParallel(model)

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	# Parameters and DataLoaders.
	input_size = 5
	output_size = 2

	batch_size = 30
	data_size = 100

	rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)

	# Create Model and DataParallel.
	model = Model(input_size, output_size)
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), 'GPUs!')
		# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs.
		model = nn.DataParallel(model)

	model.to(device)

	# Run the Model.
	for data in rand_loader:
		input = data.to(device)
		output = model(input)
		print('Outside: input size', input.size(), 'output_size', output.size())

# REF [function] >> train_distributedly() in ./pytorch_distributed.py
def data_parallel_test():
	# REF [class] >> ConvNet class in ./pytorch_distributed.py
	class ConvNet(torch.nn.Module):
		def __init__(self, num_classes=10):
			super(ConvNet, self).__init__()
			self.layer1 = torch.nn.Sequential(
				torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
				torch.nn.BatchNorm2d(16),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=2, stride=2)
			)
			self.layer2 = torch.nn.Sequential(
				torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
				torch.nn.BatchNorm2d(32),
				torch.nn.ReLU(),
				torch.nn.MaxPool2d(kernel_size=2, stride=2)
			)
			self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

		def forward(self, x):
			out = self.layer1(x)
			out = self.layer2(out)
			out = out.reshape(out.size(0), -1)
			out = self.fc(out)
			return out

	#-----
	num_epochs = 10
	batch_size = 100
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = ConvNet()
	# For data parallelism.
	if torch.cuda.device_count() > 1:
		print(f"{torch.cuda.device_count()} GPUs exist.")
		model = torch.nn.DataParallel(model)
		#model = torch.nn.DataParallel(model, device_ids=[0, 1, 2], output_device=None)
	model.to(device)

	# Define loss function (criterion) and optimizer.
	criterion = torch.nn.CrossEntropyLoss().to(device)
	optimizer = torch.optim.SGD(model.parameters(), 1e-4)

	# Load data.
	train_dataset = torchvision.datasets.MNIST(
		root='./',
		train=True,
		transform=torchvision.transforms.ToTensor(),
		download=True,
	)

	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=0,
		pin_memory=True,
	)

	start_time = datetime.datetime.now()
	total_step = len(train_loader)
	for epoch in range(num_epochs):
		for i, (images, labels) in enumerate(train_loader):
			#images = images.to(device)  # When using DP, input variables can be on any device, including CPU.
			labels = labels.to(device)

			# Forward pass.
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backward and optimize.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i + 1) % 100 == 0:
				print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}.'.format(
					epoch + 1,
					num_epochs,
					i + 1,
					total_step,
					loss.item()
				))
	print('Training complete in {}.'.format(datetime.datetime.now() - start_time))

def main():
	# Data parallelism (DP).

	#data_parallel_tutorial()
	data_parallel_test()

	# Distributed data parallelism (DDP).
	#	Refer to ./pytorch_distributed.py

	# Fully sharded data parallel (FSDP).
	#	https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
	#	https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html

	#-----
	# Model parallelism.
	#	https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

	# Pipeline parallelism.
	#	https://pytorch.org/docs/stable/pipeline.html
	#	https://pytorch.org/tutorials/intermediate/pipeline_tutorial.html
	#	https://pytorch.org/tutorials/advanced/ddp_pipeline.html
	#
	#	"GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", arXiv 2017.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
