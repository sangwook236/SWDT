#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://pytorch.org/tutorials/beginner/dist_overview.html
#	https://pytorch.org/docs/stable/distributed.html

import os, math, random
import torch
import torchvision

# Blocking point-to-point communication.
# REF [site] >> https://pytorch.org/tutorials/intermediate/dist_tuto.html
def run_blocking_p2p(rank, world_size, use_cuda=True):
	tensor = torch.zeros(1)
	if use_cuda:
		device = torch.device('cuda:{}'.format(rank))
		tensor = tensor.to(device)
	if rank == 0:
		tensor += 1
		# Send the tensor to process 1.
		torch.distributed.send(tensor=tensor, dst=1)
	else:
		# Receive tensor from process 0.
		torch.distributed.recv(tensor=tensor, src=0)
	print('Rank {} has data {}.'.format(rank, tensor[0]))

# Non-blocking point-to-point communication.
# REF [site] >> https://pytorch.org/tutorials/intermediate/dist_tuto.html
def run_nonblocking_p2p(rank, world_size, use_cuda=True):
	tensor = torch.zeros(1)
	if use_cuda:
		device = torch.device('cuda:{}'.format(rank))
		tensor = tensor.to(device)
	req = None
	if rank == 0:
		tensor += 1
		# Send the tensor to process 1.
		req = torch.distributed.isend(tensor=tensor, dst=1)
		print('Rank 0 started sending.')
	else:
		# Receive tensor from process 0.
		req = torch.distributed.irecv(tensor=tensor, src=0)
		print('Rank 1 started receiving.')
	req.wait()
	print('Rank {} has data {}.'.format(rank, tensor[0]))

# Custom scatter implementation.
# REF [site] >> https://shalab.usc.edu/writing-distributed-applications-with-pytorch/
def scatter(tensor, rank, tensor_list=None, root=0, group=None):
	""" Sends the i-th tensor in tensor_list on root to the i-th process. """
	#rank = torch.distributed.get_rank()
	if group is None:
		group = torch.distributed.group.WORLD
	if rank == root:
		assert(tensor_list is not None)
		torch.distributed.scatter_send(tensor_list, tensor, group)
	else:
		torch.distributed.scatter_recv(tensor, root, group)

# Custom gather implementation.
# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/ptp.py
def gather(tensor, rank, tensor_list=None, root=0, group=None):
	""" Sends tensor to root process, which store it in tensor_list. """
	#rank = torch.distributed.get_rank()
	if group is None:
		group = torch.distributed.group.WORLD
	if rank == root:
		assert(tensor_list is not None)
		torch.distributed.gather_recv(tensor_list, tensor, group)
	else:
		torch.distributed.gather_send(tensor, root, group)

# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/ptp.py
def run_gather_p2p(rank, world_size, use_cuda=True):
	""" Simple point-to-point communication. """
	if use_cuda:
		device = torch.device('cuda:{}'.format(rank))
		tensor = torch.ones(1).to(device)
		tensor_list = [torch.zeros(1).to(device) for _ in range(world_size)]
	else:
		tensor = torch.ones(1)
		tensor_list = [torch.zeros(1) for _ in range(world_size)]
	if rank == 0:
		torch.distributed.gather(tensor, gather_list=tensor_list, dst=0)
		#gather(tensor, rank, tensor_list=tensor_list, root=0, group=None)
		print('Gathered output = {}.'.format(tensor_list))
	else:
		torch.distributed.gather(tensor, gather_list=None, dst=0)
		#gather(tensor, rank, tensor_list=None, root=0, group=None)

	print('Rank {} has data {}.'.format(rank, sum(tensor_list)[0]))

# All-Reduce example.
# REF [site] >> https://pytorch.org/tutorials/intermediate/dist_tuto.html
def run_all_reduce_p2p(rank, world_size, use_cuda=True):
	""" Simple point-to-point communication. """
	group = torch.distributed.new_group([0, 1])
	tensor = torch.ones(1)
	if use_cuda:
		device = torch.device('cuda:{}'.format(rank))
		tensor = tensor.to(device)
	torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=group)
	print('Rank {}  has data {}.'.format(rank, tensor[0]))

# REF [site] >>
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/gloo.py
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/allreduce.py
def all_reduce(send, recv, use_cuda):
	""" Implementation of a ring-reduce. """
	rank = torch.distributed.get_rank()
	world_size = torch.distributed.get_world_size()
	if use_cuda: device = torch.device('cuda:{}'.format(rank))

	send_buff = torch.zeros(send.size())
	recv_buff = torch.zeros(send.size())
	accum = torch.zeros(send.size())
	if use_cuda: accum = accum.to(device)
	accum[:] = send[:]
	if use_cuda: torch.cuda.synchronize()

	left = ((rank - 1) + world_size) % world_size
	right = (rank + 1) % world_size

	for i in range(world_size - 1):
		if i % 2 == 0:
			# Send send_buff.
			send_req = torch.distributed.isend(send_buff, right)
			torch.distributed.recv(recv_buff, left)
			accum[:] += recv[:]
		else:
			# Send recv_buff.
			send_req = torch.distributed.isend(recv_buff, right)
			torch.distributed.recv(send_buff, left)
			accum[:] += send[:]
		send_req.wait()
	if use_cuda: torch.cuda.synchronize()
	recv[:] = accum[:]

# REF [site] >>
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/gloo.py
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/allreduce.py
def run_all_reduce(rank, world_size, use_cuda=True):
	""" Distributed function to be implemented later. """
	if use_cuda:
		device = torch.device('cuda:{}'.format(rank))
		#t = torch.ones(2, 2).to(device)
		t = torch.rand(2, 2).to(device)
	else:
		#t = torch.ones(2, 2)
		t = torch.rand(2, 2)

	#for _ in range(10000000):
	for _ in range(4):
		c = t.clone()
		#torch.distributed.all_reduce(c, torch.distributed.ReduceOp.SUM)
		all_reduce(t, c, use_cuda)
		t.set_(c)
	print(t)

# Dataset partitioning helper.
# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
class Partition(object):
	def __init__(self, data, index):
		self.data = data
		self.index = index

	def __len__(self):
		return len(self.index)

	def __getitem__(self, index):
		data_idx = self.index[index]
		return self.data[data_idx]

# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
class DataPartitioner(object):
	def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
		self.data = data
		self.partitions = []
		rng = random.Random()
		rng.seed(seed)
		data_len = len(data)
		indexes = [x for x in range(0, data_len)]
		rng.shuffle(indexes)

		for frac in sizes:
			part_len = int(frac * data_len)
			self.partitions.append(indexes[0:part_len])
			indexes = indexes[part_len:]

	def use(self, partition):
		return Partition(self.data, self.partitions[partition])

# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
class Net(torch.nn.Module):
	def __init__(self):
		super().__init__()

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

# Partitioning MNIST.
# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def partition_dataset():
	dataset = torchvision.datasets.MNIST(
		'./data', train=True, download=True,
		transform=torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize((0.1307,), (0.3081,))
		])
	)
	size = torch.distributed.get_world_size()
	batch_size = math.ceil(128 / float(size))
	partition_sizes = [1.0 / size for _ in range(size)]
	partition = DataPartitioner(dataset, partition_sizes)
	partition = partition.use(torch.distributed.get_rank())
	train_dataloader = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True)
	return train_dataloader, batch_size

# Gradient averaging.
# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def average_gradients(model):
	size = float(torch.distributed.get_world_size())
	for param in model.parameters():
		torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
		param.grad.data /= size

# Distributed synchronous SGD example.
# REF [site] >> https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
def run_synchronous_sgd(rank, world_size, use_cuda=True):
	if use_cuda: device = torch.device('cuda:{}'.format(rank))

	torch.manual_seed(1234)
	train_dataloader, batch_size = partition_dataset()

	model = Net()
	if use_cuda: model = model.to(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

	#num_batches = math.ceil(len(train_dataloader.dataset) / float(batch_size))
	num_batches = len(train_dataloader)
	for epoch in range(10):
		epoch_loss = 0.0
		for data, target in train_dataloader:
			if use_cuda:
				data, target = torch.autograd.Variable(data.to(device)), torch.autograd.Variable(target.to(device))
			else:
				data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
			
			optimizer.zero_grad()
			output = model(data)
			loss = torch.nn.functional.nll_loss(output, target)
			epoch_loss += loss.item()
			loss.backward()
			average_gradients(model)
			optimizer.step()
		print('Rank {}, epoch {}: loss = {}.'.format(torch.distributed.get_rank(), epoch, epoch_loss / num_batches))

# REF [site] >>
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/gloo.py
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/allreduce.py
def init_process(rank, world_size, use_cuda, fn, backend='gloo'):
	""" Initialize the distributed environment. """
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '29500'
	if backend in ['gloo', 'nccl']:
		#torch.distributed.init_process_group(backend, init_method=None, timeout=datetime.timedelta(0, 1800), world_size=-1, rank=-1, store=None, group_name='')
		torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
	elif backend == 'mpi':
		torch.distributed.init_process_group(backend)
	else:
		raise ValueError('Invalid backend, {}.'.format(backend))

	#print('torch.distributed.is_initialized() = {}.'.format(torch.distributed.is_initialized()))
	#print('torch.distributed.get_rank() = {}.'.format(torch.distributed.get_rank()))
	#print('torch.distributed.get_world_size() = {}.'.format(torch.distributed.get_world_size()))
	#print('torch.distributed.get_backend() = {}.'.format(torch.distributed.get_backend()))

	fn(rank, world_size, use_cuda)

# REF [site] >> 
#	https://github.com/seba-1511/dist_tuto.pth/blob/gh-pages/train_dist.py
#	https://pytorch.org/tutorials/intermediate/dist_tuto.html
def distributed_tutorial():
	torch.multiprocessing.set_start_method('spawn')

	if False:
		run_functor = run_blocking_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_nonblocking_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_gather_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_all_reduce_p2p
		use_cuda = True
	elif False:
		run_functor = run_all_reduce
		use_cuda = True
	else:
		run_functor = run_synchronous_sgd
		use_cuda = True

	backend = 'nccl' if use_cuda else 'gloo'
	world_size = 2

	processes = list()
	for rank in range(world_size):
		p = torch.multiprocessing.Process(target=init_process, args=(rank, world_size, use_cuda, run_functor, backend))
		p.start()
		processes.append(p)

	for p in processes:
		p.join()

# REF [site] >> https://pytorch.org/tutorials/intermediate/dist_tuto.html
def mpi_distributed_tutorial():
	if False:
		run_functor = run_blocking_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_nonblocking_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_gather_p2p
		use_cuda = False  # Fixed.
	elif False:
		run_functor = run_all_reduce_p2p
		use_cuda = True
	elif False:
		run_functor = run_all_reduce
		use_cuda = True
	else:
		run_functor = run_synchronous_sgd
		use_cuda = True

	init_process(0, 0, use_cuda, run_functor, backend='mpi')

def main():
	if not torch.distributed.is_available():
		print('PyTorch Distributed not available.')
		return

	#print('torch.distributed.is_mpi_available() = {}.'.format(torch.distributed.is_mpi_available()))
	#print('torch.distributed.is_nccl_available() = {}.'.format(torch.distributed.is_nccl_available()))
	#print('torch.distributed.is_initialized() = {}.'.format(torch.distributed.is_initialized()))

	#--------------------
	distributed_tutorial()

	# RuntimeError: Distributed package doesn't have MPI built in. MPI is only included if you build PyTorch from source on a host that has MPI installed.
	#mpi_distributed_tutorial()  # Use mpirun to run.

#--------------------------------------------------------------------

# Usage:
#	mpirun -np 4 python pytorch_distributed.py

if '__main__' == __name__:
	main()
