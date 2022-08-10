#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import torch
import onnxruntime

# REF [site] >> https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
def python_api_example():
	class Model(torch.nn.Module):
		def __init__(self):
			super(Model, self).__init__()

		def forward(self, x, y):
			return x.add(y)

	def create_model(type: torch.dtype = torch.float32):
		sample_x = torch.ones(3, dtype=type)
		sample_y = torch.zeros(3, dtype=type)

		torch.onnx.export(
			Model(),
			(sample_x, sample_y),
			onnx_filepath,
			input_names=['x', 'y'], output_names=['z'],
			dynamic_axes={'x': {0 : 'array_length_x'}, 'y': {0: 'array_length_y'}}
		)

	# Run the model on device consuming and producing ORTValues.
	def run_with_data_on_device(onnx_filepath: str, device_name: str, device_index: int, x: np.array, y: np.array) -> onnxruntime.OrtValue:
		providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
		session = onnxruntime.InferenceSession(onnx_filepath, providers=providers)

		x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, device_name, device_index)
		y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, device_name, device_index)

		io_binding = session.io_binding()
		io_binding.bind_input(name='x', device_type=x_ortvalue.device_name(), device_id=0, element_type=x.dtype, shape=x_ortvalue.shape(), buffer_ptr=x_ortvalue.data_ptr())
		io_binding.bind_input(name='y', device_type=y_ortvalue.device_name(), device_id=0, element_type=y.dtype, shape=y_ortvalue.shape(), buffer_ptr=y_ortvalue.data_ptr())
		io_binding.bind_output(name='z', device_type=device_name, device_id=device_index, element_type=x.dtype, shape=x_ortvalue.shape())
		session.run_with_iobinding(io_binding)

		z = io_binding.get_outputs()

		return z[0]

	# Run the model on device consuming and producing native PyTorch tensors.
	def run_with_torch_tensors_on_device(onnx_filepath: str, device_name: str, device_index: int, x: torch.Tensor, y: torch.Tensor, np_type: np.dtype = np.float32, torch_type: torch.dtype = torch.float32) -> torch.Tensor:
		providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
		session = onnxruntime.InferenceSession(onnx_filepath, providers=providers)

		binding = session.io_binding()

		x_tensor = x.contiguous()
		y_tensor = y.contiguous()

		binding.bind_input(
			name='x',
			device_type=device_name,
			device_id=device_index,
			element_type=np_type,
			shape=tuple(x_tensor.shape),
			buffer_ptr=x_tensor.data_ptr(),
		)
		binding.bind_input(
			name='y',
			device_type=device_name,
			device_id=device_index,
			element_type=np_type,
			shape=tuple(y_tensor.shape),
			buffer_ptr=y_tensor.data_ptr(),
		)

		# Allocate the PyTorch tensor for the model output.
		z_tensor = torch.empty(x_tensor.shape, dtype=torch_type, device=torch.device(f'{device_name}:{device_index}')).contiguous()
		binding.bind_output(
			name='z',
			device_type=device_name,
			device_id=device_index,
			element_type=np_type,
			shape=tuple(z_tensor.shape),
			buffer_ptr=z_tensor.data_ptr(),
		)

		session.run_with_iobinding(binding)

		return z_tensor

	#--------------------
	onnx_filepath = './model.onnx'
	create_model()

	device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
	device_index = 0
	device = torch.device(f'{device_name}:{device_index}')

	#-----
	# Run the model on CPU consuming and producing numpy arrays.
	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
	session = onnxruntime.InferenceSession(onnx_filepath, providers=providers)

	x = np.float32([1.0, 2.0, 3.0])
	y = np.float32([4.0, 5.0, 6.0])
	z = session.run(['z'], {'x': x, 'y': y})

	print('z = {}.'.format(z))  # [array([5., 7., 9.], dtype=float32)].

	#-----
	x = np.float32([1.0, 2.0, 3.0, 4.0, 5.0])
	y = np.float32([1.0, 2.0, 3.0, 4.0, 5.0])
	z = run_with_data_on_device(onnx_filepath, device_name, device_index, x, y).numpy()

	print('z = {}.'.format(z))  # [ 2.,  4.,  6.,  8., 10.].

	#-----
	x = torch.rand(5).to(device)
	y = torch.rand(5).to(device)
	z = run_with_torch_tensors_on_device(onnx_filepath, device_name, device_index, x, y)

	print('z = {}.'.format(z))  # tensor([...], device='cuda:0').

	#-----
	create_model(torch.int64)

	x = torch.ones(5, dtype=torch.int64).to(device)
	y = torch.zeros(5, dtype=torch.int64).to(device)
	z = run_with_torch_tensors_on_device(onnx_filepath, device_name, device_index, x, y, np_type=np.int64, torch_type=torch.int64)

	print('z = {}.'.format(z))  # tensor([1, 1, 1, 1, 1], device='cuda:0').

def main():
	python_api_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
