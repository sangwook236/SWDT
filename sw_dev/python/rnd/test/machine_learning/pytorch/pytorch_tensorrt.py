#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://pytorch.org/TensorRT/
#	https://github.com/pytorch/TensorRT

import copy, time
import numpy as np
import torch, torchvision
import torch_tensorrt

# REF [site] >> https://pytorch.org/TensorRT/getting_started/getting_started_with_python_api.html
def getting_started():
	class MyModel(torch.nn.Module):
		def __init__(self):
			super().__init__()

			# 1 input image channel, 6 output channels, 3x3 square convolution kernel.
			self.conv1 = torch.nn.Conv2d(1, 6, 3)
			self.conv2 = torch.nn.Conv2d(6, 16, 3)
			# An affine operation: y = Wx + b.
			self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # For 32x32 input.
			self.fc2 = torch.nn.Linear(120, 84)
			self.fc3 = torch.nn.Linear(84, 10)

		def forward(self, x):
			# Max pooling over a (2, 2) window.
			x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
			# If the size is a square you can only specify a single number.
			x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
			x = x.view(-1, self.num_flat_features(x))
			x = torch.nn.functional.relu(self.fc1(x))
			x = torch.nn.functional.relu(self.fc2(x))
			x = self.fc3(x)
			return x

		def num_flat_features(self, x):
			size = x.size()[1:]  # All dimensions except the batch dimension.
			num_features = 1
			for s in size:
				num_features *= s
			return num_features

	input_data = torch.ones((32, 1, 32, 32))

	#--------------------
	# Torch module needs to be in eval (not training) mode.
	model = MyModel().eval()

	inputs = [
		torch_tensorrt.Input(
			min_shape=[1, 1, 16, 16],
			opt_shape=[1, 1, 32, 32],
			max_shape=[1, 1, 64, 64],
			dtype=torch.half,
		)
	]
	enabled_precisions = {torch.float, torch.half}  # Run with fp16.

	trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions)

	input_data = input_data.to("cuda").half()
	result = trt_ts_module(input_data)
	print("Result: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(result.shape, result.dtype, torch.min(result), torch.max(result)))

	torch.jit.save(trt_ts_module, "./trt_ts_module.ts")

	#--------------------
	# Deployment application.
	trt_ts_module = torch.jit.load("./trt_ts_module.ts")

	input_data = input_data.to("cuda").half()
	result = trt_ts_module(input_data)
	print("Result: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(result.shape, result.dtype, torch.min(result), torch.max(result)))

# REF [site] >> https://github.com/pytorch/TensorRT/blob/master/examples/fx/torch_trt_simple_example.py
def torch_trt_simple_example():
	# Torch module needs to be in eval (not training) mode.
	model = torchvision.models.resnet18(pretrained=True).cuda().eval()
	inputs = [torch.ones((32, 3, 224, 224), device=torch.device("cuda"))]  # type: ignore[attr-defined].

	#--------------------
	# TorchScript path.
	model_ts = copy.deepcopy(model)
	inputs_ts = copy.deepcopy(inputs)

	# fp32 test.
	with torch.inference_mode():
		ref_fp32 = model_ts(*inputs_ts)

	trt_ts_module = torch_tensorrt.compile(model_ts, inputs=inputs_ts, enabled_precisions={torch.float32})
	result_fp32 = trt_ts_module(*inputs_ts)

	assert torch.nn.functional.cosine_similarity(ref_fp32.flatten(), result_fp32.flatten(), dim=0) > 0.9999

	# fp16 test.
	model_ts = model_ts.half()
	inputs_ts = [i.cuda().half() for i in inputs_ts]

	with torch.inference_mode():
		ref_fp16 = model_ts(*inputs_ts)

	trt_ts_module = torch_tensorrt.compile(model_ts, inputs=inputs_ts, enabled_precisions={torch.float16})
	result_fp16 = trt_ts_module(*inputs_ts)

	assert torch.nn.functional.cosine_similarity(ref_fp16.flatten(), result_fp16.flatten(), dim=0) > 0.99

	#--------------------
	# FX path.
	model_fx = copy.deepcopy(model)
	inputs_fx = copy.deepcopy(inputs)

	# fp32 test.
	with torch.inference_mode():
		ref_fp32 = model_fx(*inputs_fx)

	trt_fx_module = torch_tensorrt.compile(model_fx, ir="fx", inputs=inputs_fx, enabled_precisions={torch.float32})
	result_fp32 = trt_fx_module(*inputs_fx)

	assert torch.nn.functional.cosine_similarity(ref_fp32.flatten(), result_fp32.flatten(), dim=0) > 0.9999

	# fp16 test.
	model_fx = model_fx.cuda().half()
	inputs_fx = [i.cuda().half() for i in inputs_fx]

	with torch.inference_mode():
		ref_fp16 = model_fx(*inputs_fx)

	trt_fx_module = torch_tensorrt.compile(model_fx, ir="fx", inputs=inputs_fx, enabled_precisions={torch.float16})
	result_fp16 = trt_fx_module(*inputs_fx)

	assert torch.nn.functional.cosine_similarity(ref_fp16.flatten(), result_fp16.flatten(), dim=0) > 0.99

# REF [site] >> https://developer.nvidia.com/blog/accelerating-inference-up-to-6x-faster-in-pytorch-with-torch-tensorrt/
def accelerating_inference_example():
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

	#getting_started()  # Not yet tested.
	torch_trt_simple_example()  # Not yet tested.
	#accelerating_inference_example()  # Not yet tested.

#--------------------------------------------------------------------

# Install:
#	Refer to tensorrt_usage_guide.txt

if "__main__" == __name__:
	main()
