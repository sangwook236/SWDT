#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import tensorrt as trt
import transformers

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

def torch_to_onnx(onnx_filepath):
	model = Model().cuda().eval()
	input_ids = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()
	attention_mask = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()
	token_type_ids = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()

	torch.onnx.export(
		model,
		(input_ids, attention_mask, token_type_ids),
		onnx_filepath,
		input_names=['input_ids', 'attention_mask', 'token_type_ids'],
		output_names=['outputs'],
		export_params=True,
		opset_version=10
	)

def onnx_to_tensorrt(onnx_filepath, tensorrt_filepath, fp16_mode=False):
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

	builder = trt.Builder(TRT_LOGGER)
	network = builder.create_network(EXPLICIT_BATCH)
	parser = trt.OnnxParser(network, TRT_LOGGER)

	builder.max_workspace_size = (1 << 30)
	builder.fp16_mode = fp16_mode

	with open(onnx_filepath, 'rb') as fd:
		if not parser.parse(fd.read()):
			for error in range(parser.num_errors):
				print (parser.get_error(error))

	engine = builder.build_cuda_engine(network)
	buf = engine.serialize()
	with open(tensorrt_filepath, 'wb') as fd:
		fd.write(buf)

def infer_by_tensorrt(tensorrt_filepath):
	print('Start inferring by TensorRT...')
	start_total_time = time.time()
	print('Start preparing a runtime...')
	start_time = time.time()
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	trt_runtime = trt.Runtime(TRT_LOGGER)
	print('End preparing a runtime: {} secs.'.format(time.time() - start_time))

	print('Start loading a model...')
	start_time = time.time()
	with open(tensorrt_filepath, 'rb') as fd:
		engine_data = fd.read()
	engine = trt_runtime.deserialize_cuda_engine(engine_data)
	print('End loading a model: {} secs.'.format(time.time() - start_time))
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

	inputs, outputs, bindings, stream = list(), list(), list(), cuda.Stream(flags=0)
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
	context = engine.create_execution_context()

	input_ids, attention_mask, token_type_ids = np.ones([1, 512]), np.ones([1, 512]), np.ones([1, 512])

	numpy_array_input = [input_ids, attention_mask, token_type_ids]
	hosts = [inp.host for inp in inputs]
	trt_types = [trt.int32, trt.int32, trt.int32]

	for numpy_array, host, trt_type in zip(numpy_array_input, hosts, trt_types):
		numpy_array = np.asarray(numpy_array).astype(trt.nptype(trt_type)).ravel()
		np.copyto(host, numpy_array)
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	#--------------------
	print('Start inferring by the model...')
	start_time = time.time()
	def do_inference(context, bindings, inputs, outputs, stream):
		[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
		[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
		stream.synchronize()
		return [out.host for out in outputs]

	trt_outputs = do_inference(
		context=context,
		bindings=bindings,
		inputs=inputs,
		outputs=outputs,
		stream=stream
	)
	trt_outputs = trt_outputs[0].reshape((1, 512, 768))
	print('End inferring by the model: {} secs.'.format(time.time() - start_time))
	print('End inferring by TensorRT: {} secs.'.format(time.time() - start_total_time))

	#--------------------
	# For comparison.
	print('Start inferring by PyTorch...')
	start_total_time = time.time()
	print('Start loading a model...')
	start_time = time.time()
	model = Model().cuda().eval()
	print('End loading a model: {} secs.'.format(time.time() - start_time))

	print('Start preparing data...')
	start_time = time.time()
	input_ids = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()
	attention_mask = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()
	token_type_ids = torch.tensor(np.ones([1, 512]), dtype=torch.long).cuda()
	print('End preparing data: {} secs.'.format(time.time() - start_time))

	print('Start inferring by the model...')
	start_time = time.time()
	model_outputs = model(input_ids, attention_mask, attention_mask)  # (1, 512, 768).
	print('End inferring by the model: {} secs.'.format(time.time() - start_time))
	print('End inferring by PyTorch: {} secs.'.format(time.time() - start_total_time))

# REF [site] >> https://si-analytics.tistory.com/33
def transformers_example():
	fp16_mode = True
	onnx_filepath = './bert.onnx'
	if fp16_mode:
		tensorrt_filepath = './bert_fp16.plan'
	else:
		tensorrt_filepath = './bert_fp32.plan'

	#--------------------
	if False:
		print('Start converting PyTorch to ONNX...')
		start_time = time.time()
		torch_to_onnx(onnx_filepath)
		print('End converting PyTorch to ONNX: {} secs.'.format(time.time() - start_time))

		print('Start converting ONNX to TensorRT...')
		start_time = time.time()
		onnx_to_tensorrt(onnx_filepath, tensorrt_filepath, fp16_mode)
		print('End converting ONNX to TensorRT: {} secs.'.format(time.time() - start_time))

	infer_by_tensorrt(tensorrt_filepath)

def main():
	# TensorRT.
	transformers_example()

	# TRTorch.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
