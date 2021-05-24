#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
from filelock import FileLock
import torch
import torchvision
import horovod.torch as hvd

# REF [site] >>
#	https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py
#	https://horovod.readthedocs.io/en/stable/pytorch.html
def pytorch_mnist_example():
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
			return torch.nn.functional.log_softmax(x)

	def train(epoch, is_cuda, log_interval):
		model.train()
		# Horovod: set epoch to sampler for shuffling.
		train_sampler.set_epoch(epoch)
		for batch_idx, (data, target) in enumerate(train_loader):
			if is_cuda:
				data, target = data.cuda(), target.cuda()
			optimizer.zero_grad()
			output = model(data)
			loss = torch.nn.functional.nll_loss(output, target)
			loss.backward()
			optimizer.step()
			if batch_idx % log_interval == 0:
				# Horovod: use train_sampler to determine the number of examples in
				# this worker's partition.
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_sampler),
					100. * batch_idx / len(train_loader), loss.item()))

	def metric_average(val, name):
		tensor = torch.tensor(val)
		avg_tensor = hvd.allreduce(tensor, name=name)
		return avg_tensor.item()

	def test(is_cuda):
		model.eval()
		test_loss = 0.
		test_accuracy = 0.
		for data, target in test_loader:
			if is_cuda:
				data, target = data.cuda(), target.cuda()
			output = model(data)
			# sum up batch loss
			test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item()
			# get the index of the max log-probability
			pred = output.data.max(1, keepdim=True)[1]
			test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

		# Horovod: use test_sampler to determine the number of examples in
		# this worker's partition.
		test_loss /= len(test_sampler)
		test_accuracy /= len(test_sampler)

		# Horovod: average metric values across workers.
		test_loss = metric_average(test_loss, 'avg_loss')
		test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

		# Horovod: print output only on first rank.
		if hvd.rank() == 0:
			print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
				test_loss, 100. * test_accuracy))

	batch_size = 64
	test_batch_size = 1000
	epochs = 10
	lr = 0.01
	momentum = 0.5
	random_seed = 42
	log_interval = 10
	fp16_allreduce = False
	use_adasum = False
	gradient_predivide_factor = 1.0
	data_dir = './data'
	is_cuda = torch.cuda.is_available()

	# Horovod: initialize library.
	hvd.init()
	torch.manual_seed(random_seed)

	if is_cuda:
		# Horovod: pin GPU to local rank.
		torch.cuda.set_device(hvd.local_rank())
		torch.cuda.manual_seed(random_seed)

	# Horovod: limit # of CPU threads to be used per worker.
	torch.set_num_threads(1)

	kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
	# When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent issues with Infiniband implementations that are not fork-safe.
	if (kwargs.get('num_workers', 0) > 0 and hasattr(torch.multiprocessing, '_supports_context') and torch.multiprocessing._supports_context and 'forkserver' in torch.multiprocessing.get_all_start_methods()):
		kwargs['multiprocessing_context'] = 'forkserver'

	with FileLock(os.path.expanduser('~/.horovod_lock')):
		train_dataset = torchvision.datasets.MNIST(
			data_dir, train=True, download=True,
			transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081,))
			])
		)
	# Horovod: use DistributedSampler to partition the training data.
	train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, **kwargs)

	test_dataset = torchvision.datasets.MNIST(
		data_dir, train=False, download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.1307,), (0.3081,))
		])
	)
	# Horovod: use DistributedSampler to partition the test data.
	test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler, **kwargs)

	model = Net()

	# By default, Adasum doesn't need scaling up learning rate.
	lr_scaler = hvd.size() if not use_adasum else 1

	if is_cuda:
		# Move model to GPU.
		model.cuda()
		# If using GPU Adasum allreduce, scale learning rate by local_size.
		if use_adasum and hvd.nccl_built():
			lr_scaler = hvd.local_size()

	# Horovod: scale learning rate by lr_scaler.
	optimizer = torch.optim.SGD(model.parameters(), lr=lr * lr_scaler, momentum=momentum)

	# Horovod: broadcast parameters & optimizer state.
	hvd.broadcast_parameters(model.state_dict(), root_rank=0)
	hvd.broadcast_optimizer_state(optimizer, root_rank=0)

	# Horovod: (optional) compression algorithm.
	compression = hvd.Compression.fp16 if fp16_allreduce else hvd.Compression.none

	# Horovod: wrap optimizer with DistributedOptimizer.
	optimizer = hvd.DistributedOptimizer(optimizer,
		named_parameters=model.named_parameters(),
		compression=compression,
		op=hvd.Adasum if use_adasum else hvd.Average,
		gradient_predivide_factor=gradient_predivide_factor
	)

	for epoch in range(1, epochs + 1):
		train(epoch, is_cuda, log_interval)
		test(is_cuda)

def main():
	pytorch_mnist_example()

#--------------------------------------------------------------------

# Usage:
#	Run training with 1 GPU on a single machine:
#		horovodrun -np 1 -H localhost:1 python horovod_test.py
#	Run training with 4 GPUs on a single machine:
#		horovodrun -np 4 python horovod_test.py
#	Run training with 8 GPUs on two machines (4 GPUs each):
#		horovodrun -np 8 -H hostname1:4,hostname2:4 python train.py

if '__main__' == __name__:
	main()
