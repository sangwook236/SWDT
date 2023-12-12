#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
def basic_operation():
	# REF [site] >>
	#	https://pytorch.org/docs/stable/tensors.html
	#	https://pytorch.org/docs/stable/tensor_attributes.html

	x = torch.empty(5, 3)
	print('x =', x)
	print('x.shape = {}, x.dtype = {}.'.format(x.shape, x.dtype))
	#print('x =', x.data)

	x = torch.rand(2, 3)
	print('x =', x)

	x = torch.randn(2, 3)
	print('x =', x)

	x = torch.randn(2, 3)
	print('x =', x)

	x = torch.randperm(5)
	print('x =', x)

	x = torch.FloatTensor(10, 12, 3, 3)
	print('x =', x.size())
	print('x =', x.size()[:])

	#--------------------
	y = torch.zeros(2, 3)
	print('y =', y)

	y = torch.ones(2, 3)
	print('y =', y)

	y = torch.arange(0, 3, step=0.5)
	print('y =', y)

	x = torch.tensor(1, dtype=torch.int32)
	#x = torch.tensor(1, dtype=torch.int32, device='cuda:1')
	print('x =', x)

	x = torch.tensor([5.5, 3])
	print('x =', x)

	x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes.
	print('x =', x)
	x = torch.randn_like(x, dtype=torch.float) # Override dtype.
	print('x =', x)

	#--------------------
	y = torch.rand(5, 3)
	print('x + y =', x + y)

	print('x + y =', torch.add(x, y))

	result = torch.empty(5, 3)
	torch.add(x, y, out=result)
	print('x + y =', result)

	#--------------------
	# Any operation that mutates a tensor in-place is post-fixed with an _.
	# For example: x.copy_(y), x.t_(), will change x.

	y.add_(x)  # In-place.
	print('y =', y)

	#--------------------
	# You can use standard NumPy-like indexing with all bells and whistles!
	print(x[:, 1])

	#--------------------
	# If you have a one element tensor, use .item() to get the value as a Python number.
	x = torch.randn(1)
	print('x =', x)
	print('x.item() =', x.item())

	#--------------------
	x = torch.randn(2, 2)
	print('x.is_cuda =', x.is_cuda)
	print('x.is_complex() =', x.is_complex())
	print('x.is_contiguous() =', x.is_contiguous())
	print('x.is_distributed() =', x.is_distributed())
	print('x.is_floating_point() =', x.is_floating_point())
	print('x.is_pinned() =', x.is_pinned())
	print('x.is_quantized =', x.is_quantized)
	print('x.is_shared() =', x.is_shared())
	print('x.is_signed() =', x.is_signed())
	print('x.is_sparse =', x.is_sparse)

	print('x.contiguous() =', x.contiguous())
	print('x.storage() =', x.storage())

	#--------------------
	x = torch.randn(2, 2)
	print('torch.is_tensor(x) =', torch.is_tensor(x))
	print('torch.is_storage(x) =', torch.is_storage(x))
	print('torch.is_complex(x) =', torch.is_complex(x))
	print('torch.is_floating_point(x) =', torch.is_floating_point(x))

	# Sets the default floating point dtype to d.
	# This type will be used as default floating point type for type inference in torch.tensor().
	torch.set_default_dtype(torch.float32)
	print('torch.get_default_dtype() =', torch.get_default_dtype())
	# Sets the default torch.Tensor type to floating point tensor type.
	# This type will also be used as default floating point type for type inference in torch.tensor().
	torch.set_default_tensor_type(torch.FloatTensor)

	#--------------------
	# REF [site] >> https://pytorch.org/docs/stable/tensor_view.html
	# View tensor shares the same underlying data with its base tensor.
	# Supporting View avoids explicit data copy, thus allows us to do fast and memory efficient reshaping, slicing and element-wise operations.
	
	# If you want to resize/reshape tensor, you can use torch.view.
	x = torch.randn(4, 4)
	y = x.view(16)
	z = x.view(-1, 8)  # The size -1 is inferred from other dimensions.
	print('x.size() = {}, y.size() = {}, z.size() = {}.'.format(x.size(), y.size(), z.size()))

	t = torch.rand(4, 4)
	b = t.view(2, 8)
	print('t.storage().data_ptr() == b.storage().data_ptr()?', t.storage().data_ptr() == b.storage().data_ptr())

# REF [site] >>
#	https://pytorch.org/docs/stable/cuda.html
#	https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
def cuda_operation():
	print('torch.cuda.device_count() =', torch.cuda.device_count())

	print('torch.cuda.current_device() =', torch.cuda.current_device())
	torch.cuda.set_device(1)
	#torch.cuda.set_device('cuda:1')
	#torch.cuda.set_device(torch.device(1))
	#torch.cuda.set_device(torch.device('cuda:1'))
	print('torch.cuda.current_device() =', torch.cuda.current_device())

	print('torch.cuda.is_available() =', torch.cuda.is_available())
	print('torch.cuda.is_initialized() =', torch.cuda.is_initialized())
	if not torch.cuda.is_initialized():
		torch.cuda.init()
		print('torch.cuda.is_initialized() =', torch.cuda.is_initialized())

	device = None  # Uses the current device.
	#device = 1
	#device = 'cuda:1'
	print('torch.cuda.get_device_name({}) = {}.'.format(device, torch.cuda.get_device_name(device)))
	print('torch.cuda.get_device_capability({}) = {}.'.format(device, torch.cuda.get_device_capability(device)))

	#--------------------
	x = torch.tensor([1, 2, 3], dtype=torch.int32, device='cuda:1')
	#print('torch.cuda.device_of(x).idx =', torch.cuda.device_of(x).idx)

	print('x.device =', x.device)

	#--------------------
	z = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
	print('z =', z)

	print('z.cuda() =', z.cuda())
	print("z.to('cuda') =", z.to('cuda'))

	z_cuda = z.cuda()
	print('z_cuda.cpu() =', z_cuda.cpu())
	print("z_cuda.to('cpu') =", z_cuda.to('cpu'))

	#--------------------
	gpu = 0
	device = torch.device(('cuda:{}'.format(gpu) if gpu in range(torch.cuda.device_count()) else 'cuda') if torch.cuda.is_available() else 'cpu')

	x = torch.randn(2, 2)
	y = torch.ones_like(x, device=device)  # Directly create a tensor on GPU.
	x = x.to(device)  # Or just use strings .to('cuda').
	z = x + y
	print('z =', z)
	print('z =', z.to(device, torch.double))  # .to() can also change dtype together!

	#--------------------
	# Delete a tensor in a GPU.
	print(torch.cuda.memory_allocated())
	print(torch.cuda.memory_reserved())

	del x

	print(torch.cuda.memory_allocated())
	print(torch.cuda.memory_reserved())

	torch.cuda.empty_cache()

	print(torch.cuda.memory_allocated())
	print(torch.cuda.memory_reserved())

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
def numpy_bridge():
	# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.

	#--------------------
	a = torch.ones(5)
	print('a = ', a)

	b = a.numpy()
	print('b = ', b)

	# See how the numpy array changed in value.
	a.add_(1)
	print('a = ', a)
	print('b = ', b)

	#--------------------
	a = np.ones(5)
	b = torch.from_numpy(a)
	np.add(a, 1, out=a)
	print('a = ', a)
	print('b = ', b)

def parameter_test():
	if True:
		class MyModule(torch.nn.Module):
			def __init__(self, dim_input: int, dim_output: int):
				super().__init__()

				# NOTE [info] >> self.t1 is not a torch.nn.Parameter.
				self.t1 = torch.empty((dim_input, dim_output), dtype=torch.float32)  # torch.Tensor.

				self.t2 = torch.FloatTensor(dim_input, dim_output)  # torch.Tensor.
				self.p2 = torch.nn.Parameter(self.t2, requires_grad=True)  # torch.nn.Parameter.

				t3 = torch.FloatTensor(dim_input, dim_output)  # torch.Tensor.
				p3 = torch.nn.Parameter(t3, requires_grad=True)  # torch.nn.Parameter.
				# NOTE [info] >> self.p3 is accessible.
				self.register_parameter(name="p3", param=p3)

				t4 = torch.FloatTensor(dim_input, dim_output)  # torch.Tensor.
				# NOTE [info] >> self.b4 is accessible.
				self.register_buffer(name="b4", tensor=t4, persistent=True)

				# NOTE [info] >> self.l has two parameters, weigth & bias.
				self.l = torch.nn.Linear(dim_input, dim_output)  # torch.nn.Module.

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				t1 = torch.matmul(x, self.t1)
				t2 = torch.matmul(x, self.t2)
				p2 = torch.matmul(x, self.p2)
				#t3 = torch.matmul(x, self.t3)  # AttributeError: 'MyModule' object has no attribute 't3'.
				p3 = torch.matmul(x, self.p3)
				#t4 = torch.matmul(x, self.t4)  # AttributeError: 'MyModule' object has no attribute 't4'.
				b4 = torch.matmul(x, self.b4)
				l = self.l(x)
				return l

		print(f"{'-' * 20} Model:")
		dim_input, dim_output = 7, 3
		batch_size = 6

		model = MyModule(dim_input, dim_output)
		print(model)

		print(f"{'-' * 20} Parameters:")
		#for param in model.parameters(recurse=True):
		#	print(type(param), param.size())
		for name, param in model.named_parameters(prefix="", recurse=True, remove_duplicate=False):
		#for name, param in model.named_parameters(prefix="MyModule", recurse=True, remove_duplicate=False):
			print(f"{name}: {type(param)}, {param.size()}.")

		print(f"{'-' * 20} Buffers:")
		#for buf in model.buffers(recurse=True):
		#	print(type(buf), buf.size())
		for name, buf in model.named_buffers(prefix="", recurse=True, remove_duplicate=False):
		#for name, buf in model.named_buffers(prefix="MyModule", recurse=True, remove_duplicate=False):
			print(f"{name}: {type(buf)}, {buf.size()}.")

		print(f"{'-' * 20} Modules:")
		#for module in model.modules():
		#	print(type(module))
		for name, module in model.named_modules(memo=None, prefix="", remove_duplicate=True):
		#for name, module in model.named_modules(memo=None, prefix="MyModule", remove_duplicate=True):
			print(f"{name}: {type(module)}.")

		print(f"{'-' * 20} Calling the model:")
		x = torch.randn(batch_size, dim_input)
		y = model(x)
		print(f"{y.shape=}.")

	#--------------------
	if False:
		class Embedding(torch.nn.Module):
			def __init__(self, dim_input: int, dim_output: int):
				super().__init__()

				self.w = torch.nn.Linear(dim_input, dim_output)

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				return self.w(x)

		class MLP(torch.nn.Module):
			def __init__(self, dim_input: int, dim_hidden: int, dim_output: int, dropout: float = 0.1):
				super().__init__()

				self.w_1 = torch.nn.Linear(dim_input, dim_hidden)
				self.w_2 = torch.nn.Linear(dim_hidden, dim_output)
				self.dropout = torch.nn.Dropout(dropout)

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))

		class Generator(torch.nn.Module):
			def __init__(self, dim_input: int, dim_output: int):
				super().__init__()

				self.w = torch.nn.Linear(dim_input, dim_output)

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				return self.w(x)

		class MyModule(torch.nn.Module):
			def __init__(self, dim_input: int, dim_hidden: int, dim_output: int):
				super().__init__()

				self.embedding = Embedding(dim_input, dim_hidden)
				self.mlp = MLP(dim_hidden, dim_hidden // 2, dim_hidden)
				self.generator = Generator(dim_hidden, dim_output)

			def forward(self, x: torch.Tensor) -> torch.Tensor:
				x = self.embedding(x)
				x = self.mlp(x)
				x = self.generator(x)
				return x

		print(f"{'-' * 20} Model:")
		dim_input, dim_hidden, dim_output = 10, 20, 5
		batch_size = 6

		model = MyModule(dim_input, dim_hidden, dim_output)
		print(model)

		print(f"{'-' * 20} Parameters:")
		#for param in model.parameters(recurse=True):
		#	print(type(param), param.size())
		for name, param in model.named_parameters(prefix="", recurse=True, remove_duplicate=True):
		#for name, param in model.named_parameters(prefix="MyModule", recurse=True, remove_duplicate=True):
			print(f"{name}: {type(param)}, {param.size()}.")

		print(f"{'-' * 20} Buffers:")
		#for buf in model.buffers(recurse=True):
		#	print(type(buf), buf.size())
		for name, buf in model.named_buffers(prefix="", recurse=True, remove_duplicate=False):
		#for name, buf in model.named_buffers(prefix="MyModule", recurse=True, remove_duplicate=False):
			print(f"{name}: {type(buf)}, {buf.size()}.")

		print(f"{'-' * 20} Modules:")
		#for module in model.modules():
		#	print(type(module))
		for name, module in model.named_modules(memo=None, prefix="", remove_duplicate=True):
		#for name, module in model.named_modules(memo=None, prefix="MyModule", remove_duplicate=True):
			print(f"{name}: {type(module)}.")

		print(f"{'-' * 20} Calling the model:")
		x = torch.randn(batch_size, dim_input)
		y = model(x)
		print(f"{y.shape=}.")

def main():
	#basic_operation()
	#cuda_operation()

	#numpy_bridge()

	parameter_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
