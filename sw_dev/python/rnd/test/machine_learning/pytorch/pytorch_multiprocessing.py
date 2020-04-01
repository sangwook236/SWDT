#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://pytorch.org/docs/stable/notes/multiprocessing.html
#	https://pytorch.org/docs/stable/multiprocessing.html

# REF [file] >>
#	${detectron2_HOME}/tools/train_net.py
#	${detectron2_HOME}/detectron2/engine/launch.py

import os, argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision
import matplotlib.pyplot as plt

"""
# REF [site] >> https://pytorch.org/docs/stable/notes/multiprocessing.html
def train(model):
	# Construct data_loader, optimizer, etc.
	for data, labels in data_loader:
		optimizer.zero_grad()
		loss_fn(model(data), labels).backward()
		optimizer.step()  # This will update the shared parameters.

# REF [site] >> https://pytorch.org/docs/stable/notes/multiprocessing.html
def simple_example():
	num_processes = 4
	model = MyModel()
	# NOTE: this is required for the 'fork' method to work.
	model.share_memory()
	processes = []
	for rank in range(num_processes):
		p = mp.Process(target=train, args=(model,))
		p.start()
		processes.append(p)
	for p in processes:
		p.join()
"""

class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = torch.nn.Dropout2d()
		self.fc1 = torch.nn.Linear(320, 50)
		self.fc2 = torch.nn.Linear(50, 10)

	def forward(self, x):
		x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
		x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		x = x.view(-1, 320)
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.dropout(x, training=self.training)
		x = self.fc2(x)
		return torch.nn.functional.log_softmax(x, dim=1)

# REF [site] >> https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py
def train(rank, args, model, device, dataloader_kwargs):
	torch.manual_seed(args.seed + rank)

	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('../data', train=True, download=True,
			transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))
			])),
		batch_size=args.batch_size, shuffle=True, num_workers=1,
		**dataloader_kwargs)

	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
	for epoch in range(1, args.epochs + 1):
		train_epoch(epoch, args, model, device, train_loader, optimizer)

# REF [site] >> https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py
def test(args, model, device, dataloader_kwargs):
	torch.manual_seed(args.seed)

	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.MNIST('../data', train=False, transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.1307,), (0.3081,))
		])),
		batch_size=args.batch_size, shuffle=True, num_workers=1,
		**dataloader_kwargs)

	test_epoch(model, device, test_loader)

# REF [site] >> https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py
def train_epoch(epoch, args, model, device, data_loader, optimizer):
	model.train()
	pid = os.getpid()
	for batch_idx, (data, target) in enumerate(data_loader):
		optimizer.zero_grad()
		output = model(data.to(device))
		loss = torch.nn.functional.nll_loss(output, target.to(device))
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				pid, epoch, batch_idx * len(data), len(data_loader.dataset),
				100. * batch_idx / len(data_loader), loss.item()))

# REF [site] >> https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py
def test_epoch(model, device, data_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in data_loader:
			output = model(data.to(device))
			test_loss += torch.nn.functional.nll_loss(output, target.to(device), reduction='sum').item()  # Sum up batch loss.
			pred = output.max(1)[1]  # Get the index of the max log-probability.
			correct += pred.eq(target.to(device)).sum().item()

	test_loss /= len(data_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(data_loader.dataset),
		100. * correct / len(data_loader.dataset)))

# REF [site] >> https://github.com/pytorch/examples/blob/master/mnist_hogwild/main.py
def mnist_hogwild_example():
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
		help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
		help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=10, metavar='N',
		help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
		help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
		help='SGD momentum (default: 0.5)')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
		help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
		help='how many batches to wait before logging training status')
	parser.add_argument('--num-processes', type=int, default=2, metavar='N',
		help='how many training processes to use (default: 2)')
	parser.add_argument('--cuda', action='store_true', default=False,
		help='enables CUDA training')

	args = parser.parse_args()

	use_cuda = args.cuda and torch.cuda.is_available()
	device = torch.device('cuda' if use_cuda else 'cpu')
	dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

	torch.manual_seed(args.seed)
	mp.set_start_method('spawn')

	model = Net().to(device)
	model.share_memory()  # Gradients are allocated lazily, so they are not shared here.

	processes = []
	for rank in range(args.num_processes):
		p = mp.Process(target=train, args=(rank, args, model, device, dataloader_kwargs))
		# We first train the model across 'num_processes' processes.
		p.start()
		processes.append(p)
	for p in processes:
		p.join()

	# Once training is complete, we can test the model.
	test(args, model, device, dataloader_kwargs)

def main():
	#simple_example()  # Not working: a kind of skeleton code.

	mnist_hogwild_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
