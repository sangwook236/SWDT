#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		# 1 input image channel, 6 output channels, 3x3 square convolution kernel.
		self.conv1 = nn.Conv2d(1, 6, 3)
		self.conv2 = nn.Conv2d(6, 16, 3)
		# An affine operation: y = Wx + b.
		self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension.
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Max pooling over a (2, 2) window.
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number.
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # All dimensions except the batch dimension.
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
def lenet_example():
	net = Net()
	print('net =', net)

	# You just have to define the forward function, and the backward function (where gradients are computed) is automatically defined for you using autograd.
	# You can use any of the Tensor operations in the forward function.

	# The learnable parameters of a model are returned by net.parameters()
	params = list(net.parameters())
	print(len(params))
	print(params[0].size())  # conv1's .weight.

	if False:
		#--------------------
		input = torch.randn(1, 1, 32, 32)  # 32x32 input.
		out = net(input)
		print('out =', out)

		#--------------------
		# torch.nn only supports mini-batches.
		# The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
		# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
		# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.
		
		# Zero the gradient buffers of all parameters and backprops with random gradients.
		net.zero_grad()
		out.backward(torch.randn(1, 10))

	#--------------------
	# Loss.
	
	input = torch.randn(1, 1, 32, 32)  # 32x32 input.
	output = net(input)
	target = torch.randn(10)  # A dummy target, for example.
	target = target.view(1, -1)  # Make it the same shape as output.
	criterion = nn.MSELoss()

	loss = criterion(output, target)
	print('loss =', loss)

	print(loss.grad_fn)  # MSELoss.
	print(loss.grad_fn.next_functions[0][0])  # Linear.
	print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU.

	#--------------------
	# Back propagation.

	# To backpropagate the error all we have to do is to loss.backward().
	# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
	net.zero_grad()  # zeroes the gradient buffers of all parameters.

	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)

	loss.backward()

	print('conv1.bias.grad after backward')
	print(net.conv1.bias.grad)

	#--------------------
	# Update the weights.

	if False:
		# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD).
		#	weight = weight - learning_rate * gradient.
		learning_rate = 0.01
		for f in net.parameters():
			f.data.sub_(f.grad.data * learning_rate)

	# Create your optimizer.
	optimizer = optim.SGD(net.parameters(), lr=0.01)

	# In your training loop.
	for _ in range(5):
		optimizer.zero_grad()  # Zero the gradient buffers.
		output = net(input)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()  # Does the update.

def main():
	lenet_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
