#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://pytorch.org/TensorRT/
#	https://github.com/pytorch/TensorRT

import time
import numpy as np
import torch, torchvision
import torch_tensorrt

# REF [site] >> https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
def basic_example():
	torch.backends.cudnn.benchmark = True

	def benchmark(model, input_shape=(1024, 3, 512, 512), dtype="fp32", nwarmup=50, nruns=1000, device="cuda"):
		input_data = torch.randn(input_shape)
		input_data = input_data.to(device)
		if dtype == "fp16":
			input_data = input_data.half()

		print("Warm up...")
		with torch.no_grad():
			for _ in range(nwarmup):
				features = model(input_data)
		torch.cuda.synchronize()
		print("Start timing...")
		timings = []
		with torch.no_grad():
			for i in range(1, nruns + 1):
				start_time = time.time()
				model_outputs = model(input_data)
				torch.cuda.synchronize()
				end_time = time.time()
				timings.append(end_time - start_time)
				if i % 10 == 0:
					print("Iteration %d/%d, avg batch time %.2f ms" % (i, nruns, np.mean(timings) * 1000))

		print("Input shape:", input_data.size())
		print("Average throughput: %.2f images/second" % (input_shape[0] / np.mean(timings)))

	input_shape = 1, 3, 224, 224
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#--------------------
	# An instance of your model.
	model = torchvision.models.resnet18()
	model = model.eval().to(device)

	#model_outputs = model(torch.randn(input_shape).to(device))  # Run inference.

	benchmark(model, input_shape=input_shape, nruns=100, device=device)

	#--------------------
	# An example input you would normally provide to your model's forward() method.
	dummy_input = torch.randn(input_shape)
	dummy_input = dummy_input.to(device)

	# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
	traced_model = torch.jit.trace(model, dummy_input)  # torch.jit.TopLevelTracedModule.
	traced_model = traced_model.to(device)

	#model_outputs = traced_model(torch.randn(input_shape).to(device))  # Run inference.
	#torch.jit.save(traced_model, "./resnet18_traced.pth")

	benchmark(traced_model, input_shape=input_shape, nruns=100, device=device)

	#--------------------
	if True:
		trt_model = torch_tensorrt.compile(
			model, 
			inputs= [torch_tensorrt.Input(input_shape)],
			enabled_precisions= {torch_tensorrt.dtype.half}  # Run with FP16.
		)
	elif False:
		trt_model = torch_tensorrt.compile(
			model,
			inputs = [dummy_input],  # Provide example tensor for input shape.
			enabled_precisions = {torch.half},  # Run with FP16.
		)
	elif False:
		trt_model = torch_tensorrt.compile(
			model,
			inputs = [
				torch_tensorrt.Input(  # Specify input object with shape and dtype.
					min_shape=[1, 3, 224, 224],
					opt_shape=[1, 3, 512, 512],
					max_shape=[1, 3, 1024, 1024],  # For static size shape=[1, 3, 224, 224].
					dtype=torch.half  # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool).
				)
			],
			enabled_precisions = {torch.half},  # Run with FP16.
		)

	#model_outputs = trt_model(torch.randn(input_shape).to(device))  # Run inference.
	#torch.jit.save(trt_model, "./resnet18_trt.pth")

	benchmark(trt_model, input_shape=input_shape, nruns=100, dtype="fp16", device=device)

def main():
	# REF [file] >> ./pytorch_torch_script.py

	basic_example()  # Not yet tested.

#--------------------------------------------------------------------

# Install:
#	apt search libnvinfer
#	sudo apt install libnvinfer-dev=8.2.4-1+cuda11.6 libnvinfer-plugin-dev=8.2.4-1+cuda11.6
#	pip install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases

if "__main__" == __name__:
	main()
