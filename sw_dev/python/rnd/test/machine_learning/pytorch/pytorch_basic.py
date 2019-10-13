#!/usr/bin/env python

import numpy as np
import torch

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
def basic_operation():
	x = torch.empty(5, 3)
	print('x =', x)
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
	# If you want to resize/reshape tensor, you can use torch.view.
	x = torch.randn(4, 4)
	y = x.view(16)
	z = x.view(-1, 8)  # The size -1 is inferred from other dimensions.
	print(x.size(), y.size(), z.size())

	#--------------------
	# If you have a one element tensor, use .item() to get the value as a Python number.
	x = torch.randn(1)
	print('x =', x)
	print('x.item() =', x.item())

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

# REF [site] >> https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
def cuda():
	torch.cuda.is_available()

	#--------------------
	z = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
	print('z =', z)

	z_gpu = z.cuda()
	print('z =', z)

	z_cpu = z_gpu.cpu()
	print('z =', z)

	if torch.cuda.is_available():
		device = torch.device('cuda')  # A CUDA device object.
		y = torch.ones_like(x, device=device)  # Directly create a tensor on GPU.
		x = x.to(device)  # Or just use strings .to('cuda').
		z = x + y
		print('z =', z)
		print('z =', z.to('cpu', torch.double))  # .to() can also change dtype together!

def main():
	basic_operation()

	numpy_bridge()
	cuda()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
