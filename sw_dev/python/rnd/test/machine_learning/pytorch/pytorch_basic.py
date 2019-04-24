#!/usr/bin/env python

import torch

# REF [site] >> https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/01_DL%26Pytorch/2_pytorch_tensor_basic.ipynb
def basic_operation():
	x = torch.empty(5, 3)
	print('x =', x)

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

	#--------------------
	z = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
	print('z =', z)

	z_gpu = z.cuda()
	print('z =', z)

	z_cpu = z_gpu.cpu()
	print('z =', z)

def main():
	basic_operation()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
