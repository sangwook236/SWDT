#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, math, random, datetime, tempfile
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
	# Which backend to use? (rule of thumb):
	#	Use the NCCL backend for distributed GPU training.
	#	Use the Gloo backend for distributed CPU training.
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

def setup(rank, world_size):
	# On Windows platform, the torch.distributed package only supports Gloo backend, FileStore and TcpStore.
	# For FileStore, set init_method parameter in init_process_group to a local file. Example as follow:
	# init_method="file:///f:/libtmp/some_file"
	# torch.distributed.init_process_group(
	#    "gloo",
	#    rank=rank,
	#    init_method=init_method,
	#    world_size=world_size)
	# For TcpStore, same way as on Linux.

	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"

	# Initialize the process group
	#torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
	torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
	torch.distributed.destroy_process_group()

class ToyModel(torch.nn.Module):
	def __init__(self):
		super(ToyModel, self).__init__()
		self.net1 = torch.nn.Linear(10, 10)
		self.relu = torch.nn.ReLU()
		self.net2 = torch.nn.Linear(10, 5)

	def forward(self, x):
		return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size):
	print(f"Running basic DDP example on rank {rank}.")
	setup(rank, world_size)

	# Create model and move it to GPU with id rank
	model = ToyModel().to(rank)
	ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

	optimizer.zero_grad()
	outputs = ddp_model(torch.randn(20, 10))
	labels = torch.randn(20, 5).to(rank)
	loss_fn(outputs, labels).backward()
	optimizer.step()

	cleanup()

def demo_checkpoint(rank, world_size):
	print(f"Running DDP checkpoint example on rank {rank}.")
	setup(rank, world_size)

	model = ToyModel().to(rank)
	ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

	CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
	if rank == 0:
		# All processes should see same parameters as they all start from same
		# random parameters and gradients are synchronized in backward passes.
		# Therefore, saving it in one process is sufficient.
		torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

	# Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
	torch.distributed.barrier()
	# Configure map_location properly
	map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
	ddp_model.load_state_dict(
		torch.load(CHECKPOINT_PATH, map_location=map_location))

	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

	optimizer.zero_grad()
	outputs = ddp_model(torch.randn(20, 10))
	labels = torch.randn(20, 5).to(rank)
	loss_fn(outputs, labels).backward()
	optimizer.step()

	# Not necessary to use a torch.distributed.barrier() to guard the file deletion below
	# as the AllReduce ops in the backward pass of DDP already served as a synchronization.

	if rank == 0:
		os.remove(CHECKPOINT_PATH)

	cleanup()

# Combining DDP with Model Parallelism
class ToyMpModel(torch.nn.Module):
	def __init__(self, dev0, dev1):
		super(ToyMpModel, self).__init__()
		self.dev0 = dev0
		self.dev1 = dev1
		self.net1 = torch.nn.Linear(10, 10).to(dev0)
		self.relu = torch.nn.ReLU()
		self.net2 = torch.nn.Linear(10, 5).to(dev1)

	def forward(self, x):
		x = x.to(self.dev0)
		x = self.relu(self.net1(x))
		x = x.to(self.dev1)
		return self.net2(x)

def demo_model_parallel(rank, world_size):
	print(f"Running DDP with model parallel example on rank {rank}.")
	setup(rank, world_size)

	# Setup mp_model and devices for this process
	dev0 = rank * 2
	dev1 = rank * 2 + 1
	mp_model = ToyMpModel(dev0, dev1)
	ddp_mp_model = torch.nn.parallel.DistributedDataParallel(mp_model)

	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(ddp_mp_model.parameters(), lr=0.001)

	optimizer.zero_grad()
	# Outputs will be on dev1
	outputs = ddp_mp_model(torch.randn(20, 10))
	labels = torch.randn(20, 5).to(dev1)
	loss_fn(outputs, labels).backward()
	optimizer.step()

	cleanup()

# REF [site] >> https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def ddp_tutorial():
	# Now, let's create a toy module, wrap it with DDP, and feed it some dummy input data.
	# Please note, as DDP broadcasts model states from rank 0 process to all other processes in the DDP constructor,
	# you do not need to worry about different DDP processes starting from different initial model parameter values.

	def run_demo(demo_fn, world_size):
		torch.multiprocessing.spawn(
			demo_fn,
			args=(world_size,),
			nprocs=world_size,
			join=True,
		)

	#-----
	n_gpus = torch.cuda.device_count()
	assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
	print(f"Found {n_gpus} GPUs.")

	world_size = n_gpus
	run_demo(demo_basic, world_size)
	run_demo(demo_checkpoint, world_size)
	world_size = n_gpus // 2
	run_demo(demo_model_parallel, world_size)

class ConvNet(torch.nn.Module):
	def __init__(self, num_classes=10):
		super(ConvNet, self).__init__()
		self.layer1 = torch.nn.Sequential(
			torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
			torch.nn.BatchNorm2d(16),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.layer2 = torch.nn.Sequential(
			torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
		)
		self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.fc(out)
		return out

def train_distributedly(gpu, config):
	rank = config['nr'] * config['gpus'] + gpu
	torch.distributed.init_process_group(
		backend='nccl',
		init_method='env://',
		world_size=config['world_size'],
		rank=rank,
	)

	torch.manual_seed(0)
	torch.cuda.set_device(gpu)

	model = ConvNet()
	model.cuda(gpu)

	# Define loss function (criterion) and optimizer.
	criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
	optimizer = torch.optim.SGD(model.parameters(), 1e-4)

	# Wrapper around our model to handle parallel training.
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

	# Data loading code.
	train_dataset = torchvision.datasets.MNIST(
		root='./',
		train=True,
		transform=torchvision.transforms.ToTensor(),
		download=True,
	)
	
	# Sampler that takes care of the distribution of the batches such that
	# the data is not repeated in the iteration and sampled accordingly.
	train_sampler = torch.utils.data.distributed.DistributedSampler(
		train_dataset,
		num_replicas=config['world_size'],
		rank=rank,
	)
	
	# We pass in the train_sampler which can be used by the DataLoader.
	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=config['batch_size'],
		shuffle=False,
		num_workers=0,
		pin_memory=True,
		sampler=train_sampler,
	)

	start_time = datetime.datetime.now()
	total_step = len(train_loader)
	for epoch in range(config['epochs']):
		for i, (images, labels) in enumerate(train_loader):
			images = images.cuda(non_blocking=True)
			labels = labels.cuda(non_blocking=True)

			# Forward pass.
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backward and optimize.
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if (i + 1) % 100 == 0 and gpu == 0:
				print(f"Epoch [{epoch + 1}/{config['epochs']}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}.")
	if gpu == 0:
		print('Training complete in: ' + str(datetime.datetime.now() - start_time))

# REF [site] >> https://medium.com/analytics-vidhya/distributed-training-in-pytorch-part-1-distributed-data-parallel-ae5c645e74cb
def distributed_data_parallel_example():
	config = {}
	config['nodes'] = 1
	config['gpus'] = 2  # Number of gpus per node.
	config['nr'] = 0  # Ranking within the nodes.
	config['epochs'] = 2  # Number of total epochs to run.
	config['batch_size'] = 100
	config['world_size'] = config['gpus'] * config['nodes']

	os.environ['MASTER_ADDR'] = '192.168.1.3'
	os.environ['MASTER_PORT'] = '8888'
	torch.multiprocessing.spawn(train_distributedly, args=(config,), nprocs=config['gpus'])

def main():
	# REF [site] >>
	#	https://pytorch.org/docs/stable/distributed.html
	#	https://pytorch.org/tutorials/distributed/home.html
	#	https://pytorch.org/tutorials/beginner/dist_overview.html
	#	https://pytorch.org/tutorials/intermediate/dist_tuto.html

	if not torch.distributed.is_available():
		print('PyTorch Distributed not available.')
		return

	#print(f'torch.distributed.is_initialized() = {torch.distributed.is_initialized()}.')  # Check if the default process group has been initialized.
	#print(f'torch.distributed.is_mpi_available() = {torch.distributed.is_mpi_available()}.')
	#print(f'torch.distributed.is_nccl_available() = {torch.distributed.is_nccl_available()}.')
	#print(f'torch.distributed.is_gloo_available() = {torch.distributed.is_gloo_available()}.')
	#print(f'torch.distributed.is_torchelastic_launched() = {torch.distributed.is_torchelastic_launched()}.')

	#--------------------
	#distributed_tutorial()

	# When using mpirun.
	# NOTE [error] >> RuntimeError: Distributed package doesn't have MPI built in. MPI is only included if you build PyTorch from source on a host that has MPI installed.
	#mpi_distributed_tutorial()

	#--------------------
	# Data parallelism (DP).
	#	Refer to ./pytorch_data_and_model_parallelism.py

	#--------------------
	# Distributed data parallelism (DDP).
	#	https://pytorch.org/tutorials/beginner/ddp_series_intro.html
	#	https://github.com/pytorch/examples/tree/main/distributed/ddp
	#	https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series

	# https://github.com/pytorch/examples/tree/main/distributed/ddp
	#	(global) world size.
	#	Local world size.
	#	(global) rank.
	#	Local rank.

	ddp_tutorial()
	#distributed_data_parallel_example()

#--------------------------------------------------------------------

# Usage:
#	mpirun -np 4 python pytorch_distributed.py

if '__main__' == __name__:
	main()
