#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, time, json
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torchvision
import tensorrt as trt
from PIL import Image
import matplotlib.pyplot as plt
import transformers

def cuda_cnn_benchmark(model, input_shape, is_fp16=False, nwarmup=50, nruns=10000, device='cuda'):
	torch.backends.cudnn.benchmark = True

	input_data = torch.randn(input_shape, device=device)
	if is_fp16:
		input_data = input_data.half()
		
	print('Warm up...')
	with torch.no_grad():
		for _ in range(nwarmup):
			features = model(input_data)

	torch.cuda.synchronize()
	print('Start timing...')
	timings = list()
	with torch.no_grad():
		for i in range(1, nruns + 1):
			start_time = time.time()
			features = model(input_data)
			torch.cuda.synchronize()
			end_time = time.time()
			timings.append(end_time - start_time)
			if i % 100 == 0:
				print('Iteration {}/{}: Average batch time = {:.2f} msec.'.format(i, nruns, np.mean(timings) * 1000))

	print('Input shape = {}.'.format(input_data.shape))
	print('Output feature shape = {}.'.format(features.shape))
	print('Average batch time = {:.2f} msec.'.format(np.mean(timings) * 1000))

def cuda_transformer_benchmark(model, input1_shape, input2_shape, input3_shape, is_fp16=False, nwarmup=50, nruns=10000, device='cuda'):
	torch.backends.cudnn.benchmark = True

	input1_data = torch.ones(input1_shape, dtype=torch.long).to(device)
	input2_data = torch.ones(input2_shape, dtype=torch.long).to(device)
	input3_data = torch.ones(input3_shape, dtype=torch.long).to(device)
	if is_fp16:
		input1_data = input1_data.half()
		input2_data = input2_data.half()
		input3_data = input3_data.half()

	print('Warm up...')
	with torch.no_grad():
		for _ in range(nwarmup):
			features = model(input1_data, input2_data, input3_data)

	torch.cuda.synchronize()
	print('Start timing...')
	timings = list()
	with torch.no_grad():
		for i in range(1, nruns + 1):
			start_time = time.time()
			features = model(input1_data, input2_data, input3_data)
			torch.cuda.synchronize()
			end_time = time.time()
			timings.append(end_time - start_time)
			if i % 100 == 0:
				print('Iteration {}/{}: Average batch time = {:.2f} msec.'.format(i, nruns, np.mean(timings) * 1000))

	print('Input1 shape = {}, Input2 shape = {}, Input3 shape = {}.'.format(input1_data.shape, input2_data.shape, input3_data.shape))
	print('Output feature shape = {}.'.format(features.shape))
	print('Average batch time = {:.2f} msec.'.format(np.mean(timings) * 1000))

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.model = transformers.BertModel.from_pretrained('bert-base-cased')

	def forward(self, input_ids, attention_mask, token_type_ids):
		'''
		input: 
			input_ids: torch.tensor(x, dtype=torch.long), shape = (1, 512).
			attention_mask: torch.tensor(x, dtype=torch.long), shape = (1, 512).
			token_type_ids: torch.tensor(x, dtype=torch.long), shape = (1, 512).
		output:
			hiddens[-1]: torch.tensor(x, dtype=torch.float32), shape = (1, 512, 768).
		'''

		last_hiddens, last_pooling_hiddens, hiddens = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			output_hidden_states=True
		)

		return hiddens[-1]

def torch_to_onnx(model, onnx_filepath, input1_shape, input2_shape, input3_shape, device='cuda'):
	input_ids = torch.ones(input1_shape, dtype=torch.long, device=device)
	attention_mask = torch.ones(input2_shape, dtype=torch.long, device=device)
	token_type_ids = torch.ones(input3_shape, dtype=torch.long, device=device)

	print('Start exporting a PyTorch model to an ONNX file, {}...'.format(onnx_filepath))
	start_time = time.time()
	torch.onnx.export(
		model,
		(input_ids, attention_mask, token_type_ids),
		onnx_filepath,
		input_names=['input_ids', 'attention_mask', 'token_type_ids'],
		output_names=['outputs'],
		export_params=True,
		opset_version=10,
		#do_constant_folding=True,  # Constant-folding optimization.
		#dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Specify dynamic axes of input/output.
		verbose=True
	)
	print('End exporting a PyTorch model to an ONNX file: {} secs.'.format(time.time() - start_time))

	#--------------------
	import onnx

	onnx_model = onnx.load(onnx_filepath)
	onnx.checker.check_model(onnx_model)
	#print('ONNX graph:\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

	# Visualize an ONNX graph using netron.

def onnx_to_tensorrt(onnx_filepath, tensorrt_filepath, is_fp16=False):
	trt_logger = trt.Logger(trt.Logger.WARNING)
	explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

	builder = trt.Builder(trt_logger)
	network = builder.create_network(explicit_batch)
	parser = trt.OnnxParser(network, trt_logger)

	builder.max_workspace_size = (1 << 30)
	#builder.max_batch_size = 1
	if builder.platform_has_fast_fp16 and is_fp16:
		builder.fp16_mode = True

	print('Start loading and parsing an ONNX model from {}...'.format(onnx_filepath))
	start_time = time.time()
	with open(onnx_filepath, 'rb') as fd:
		if parser.parse(fd.read()):
			print('ONNX parsing succeeded.')
		else:
			print('ONNX parsing failed.')
			for error in range(parser.num_errors):
				print (parser.get_error(error))
	print('End loading and parsing an ONNX model: {} secs.'.format(time.time() - start_time))

	engine = builder.build_cuda_engine(network)
	assert engine

	print('Start exporting to a TensorRT file, {}...'.format(tensorrt_filepath))
	start_time = time.time()
	buf = engine.serialize()
	with open(tensorrt_filepath, 'wb') as fd:
		fd.write(buf)
	print('End exporting to a TensorRT file: {} secs.'.format(time.time() - start_time))

def infer_by_tensorrt(tensorrt_filepath, input1_shape, input2_shape, input3_shape, device='cuda'):
	print('Start creating a runtime...')
	start_time = time.time()
	trt_logger = trt.Logger(trt.Logger.WARNING)
	trt_runtime = trt.Runtime(trt_logger)
	print('End creating a runtime: {} secs.'.format(time.time() - start_time))

	print('Start loading a model...')
	start_time = time.time()
	with open(tensorrt_filepath, 'rb') as fd:
		engine_data = fd.read()
	print('End loading a model: {} secs.'.format(time.time() - start_time))

	engine = trt_runtime.deserialize_cuda_engine(engine_data)
	assert engine

	#--------------------
	print('Start preparing data...')
	start_time = time.time()
	class HostDeviceMem(object):
		def __init__(self, host_mem, device_mem):
			self.host = host_mem
			self.device = device_mem

		def __str__(self):
			return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

		def __repr__(self):
			return self.__str__()

	stream = cuda.Stream(flags=0)
	inputs, outputs, bindings = list(), list(), list()
	for binding in engine:
		size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
		dtype = trt.nptype(engine.get_binding_dtype(binding))
		host_mem = cuda.pagelocked_empty(size, dtype)
		device_mem = cuda.mem_alloc(host_mem.nbytes)
		bindings.append(int(device_mem))
		if engine.binding_is_input(binding):
			inputs.append(HostDeviceMem(host_mem, device_mem))
		else:
			outputs.append(HostDeviceMem(host_mem, device_mem))

	input_ids, attention_mask, token_type_ids = np.ones(input1_shape), np.ones(input2_shape), np.ones(input3_shape)

	numpy_array_input = [input_ids, attention_mask, token_type_ids]
	hosts = [inp.host for inp in inputs]
	trt_types = [trt.int32, trt.int32, trt.int32]

	for numpy_array, host, trt_type in zip(numpy_array_input, hosts, trt_types):
		numpy_array = np.asarray(numpy_array).astype(trt.nptype(trt_type)).ravel()
		np.copyto(host, numpy_array)
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	#--------------------
	def do_inference(context, bindings, inputs, outputs, stream):
		[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
		[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
		stream.synchronize()
		return [out.host for out in outputs]

	print('Start inferring by the model...')
	start_time = time.time()
	context = engine.create_execution_context()

	trt_outputs = do_inference(
		context=context,
		bindings=bindings,
		inputs=inputs,
		outputs=outputs,
		stream=stream
	)
	trt_outputs = trt_outputs[0].reshape(input1_shape + (768,))
	print('End inferring by the model: {} secs.'.format(time.time() - start_time))

def infer_by_pytorch(model, input1_shape, input2_shape, input3_shape, device='cuda'):
	input_ids = torch.tensor(np.ones(input1_shape), dtype=torch.long).to(device)
	attention_mask = torch.tensor(np.ones(input2_shape), dtype=torch.long).to(device)
	token_type_ids = torch.tensor(np.ones(input3_shape), dtype=torch.long).to(device)

	print('Start inferring by the model...')
	start_time = time.time()
	model_outputs = model(input_ids, attention_mask, attention_mask)
	print('End inferring by the model: {} secs.'.format(time.time() - start_time))

# REF [site] >> https://si-analytics.tistory.com/33
def simple_tensorrt_example(device='cuda'):
	input1_shape = input2_shape = input3_shape = 1, 512
	is_fp16 = True
	is_forced = False

	onnx_filepath = './bert.onnx'
	if is_fp16:
		tensorrt_filepath = './bert_fp16.plan'
	else:
		tensorrt_filepath = './bert_fp32.plan'

	#--------------------
	if is_forced or not os.path.exists(onnx_filepath):
		model = Model().eval().to(device)

		print('Start converting PyTorch to ONNX...')
		start_time = time.time()
		torch_to_onnx(model, onnx_filepath, input1_shape, input2_shape, input3_shape, device)
		print('End converting PyTorch to ONNX: {} secs.'.format(time.time() - start_time))

	if is_forced or not os.path.exists(tensorrt_filepath):
		print('Start converting ONNX to TensorRT...')
		start_time = time.time()
		onnx_to_tensorrt(onnx_filepath, tensorrt_filepath, is_fp16)
		print('End converting ONNX to TensorRT: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start inferring by TensorRT...')
	start_time = time.time()
	infer_by_tensorrt(tensorrt_filepath, input1_shape, input2_shape, input3_shape, device)
	print('End inferring by TensorRT: {} secs.'.format(time.time() - start_time))

	print('Start inferring by PyTorch...')
	model = Model().eval().to(device)

	start_time = time.time()
	infer_by_pytorch(model, input1_shape, input2_shape, input3_shape, device)
	print('End inferring by PyTorch: {} secs.'.format(time.time() - start_time))

	#--------------------
	cuda_transformer_benchmark(model, input1_shape, input2_shape, input3_shape, nruns=1000, device=device)

def resnet50_preprocess():
	preprocess = torchvision.transforms.Compose([
		torchvision.transforms.Resize(256),
		torchvision.transforms.CenterCrop(224),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return preprocess

# REF [site] >> https://github.com/NVIDIA/TRTorch/blob/master/notebooks/Resnet50-example.ipynb
def simple_trtorch_example(device='cuda'):
	input_shape = 128, 3, 224, 224  # (batch size, channel, height, width).

	preprocess = resnet50_preprocess()
	fig, axes = plt.subplots(nrows=2, ncols=2)
	for i in range(4):
		img_path = './data/img{}.JPG'.format(i)
		img = Image.open(img_path)
		input_tensor = preprocess(img)      
		plt.subplot(2, 2, i + 1)
		plt.imshow(img)
		plt.axis('off')

	with open('./data/imagenet_class_index.json') as fd: 
		d = json.load(fd)

	print('Number of classes in ImageNet: {}'.format(len(d)))

	#--------------------
	resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
	resnet50_model.eval()

	# Decode the results into ([predicted class, description], probability).
	def predict(img_path, model):
		img = Image.open(img_path)
		#preprocess = resnet50_preprocess()
		input_tensor = preprocess(img)
		input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model.

		# Move the input and model to GPU for speed if available.
		if torch.cuda.is_available():
			input_batch = input_batch.to(device)
			model.to(device)

		with torch.no_grad():
			output = model(input_batch)
			# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes.
			sm_output = torch.nn.functional.softmax(output[0], dim=0)

		ind = torch.argmax(sm_output)
		return d[str(ind.item())], sm_output[ind]  # ([predicted class, description], probability).

	for i in range(4):
		img_path = './data/img{}.JPG'.format(i)
		img = Image.open(img_path)

		pred, prob = predict(img_path, resnet50_model)
		print('{}: Predicted = {}, Probablility = {}.'.format(img_path, pred, prob))

		plt.subplot(2, 2, i + 1)
		plt.imshow(img)
		plt.axis('off')
		plt.title(pred[1])

	#--------------------
	# Model benchmark without TRTorch/TensorRT

	model = resnet50_model.eval().to(device)

	cuda_cnn_benchmark(model, input_shape, nruns=1000, device=device)

	#--------------------
	# Create TorchScript modules.

	# Tracing.
	model = resnet50_model.eval().to(device)
	print('Start creating a traced model...')
	start_time = time.time()
	traced_model = torch.jit.trace(model, [torch.randn(input_shape, device=device)])
	print('End creating a traced model: {} secs.'.format(time.time() - start_time))

	#torch.jit.save(traced_model, './resnet50_traced.jit.pt')

	cuda_cnn_benchmark(traced_model, input_shape, nruns=1000, device=device)

	# Compile with TRTorch.
	import trtorch

	# FP32 (single precision).
	# The compiled module will have precision as specified by "op_precision".
	print('Start creating a TRTorch model (FP32)...')
	start_time = time.time()
	trt_ts_model_fp32 = trtorch.compile(traced_model, {
		'input_shapes': [input_shape],
		'op_precision': torch.float32,  # Run with FP32.
		'workspace_size': 1 << 20
	})
	print('End creating a TRTorch model (FP32): {} secs.'.format(time.time() - start_time))

	#torch.jit.save(trt_ts_model_fp32, './resnet50_fp32.ts')

	cuda_cnn_benchmark(trt_ts_model_fp32, input_shape, nruns=1000, device=device)

	# FP16 (half precision).
	# The compiled module will have precision as specified by "op_precision".
	print('Start creating a TRTorch model (FP16)...')
	start_time = time.time()
	trt_ts_model_fp16 = trtorch.compile(traced_model, {
		'input_shapes': [input_shape],
		'op_precision': torch.half,  # Run with FP16.
		'workspace_size': 1 << 20
	})
	print('End creating a TRTorch model (FP16): {} secs.'.format(time.time() - start_time))

	#torch.jit.save(trt_ts_model_fp16, './resnet50_fp16.ts')

	cuda_cnn_benchmark(trt_ts_model_fp16, input_shape, is_fp16=True, nruns=1000, device=device)

def main():
	device = torch.device('cuda')

	# TensorRT
	simple_tensorrt_example(device)

	# TRTorch
	#simple_trtorch_example(device)

	# Segment Anything (SAM)
	#	Refer to sam_tensorrt_test.py

#--------------------------------------------------------------------

# Usage:
#	CUDA_VISIBLE_DEVICES=1 python tensorrt_test.py

if '__main__' == __name__:
	main()
