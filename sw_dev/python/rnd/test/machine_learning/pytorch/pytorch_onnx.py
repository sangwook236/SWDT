#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch, torchvision
import onnx

# REF [site] >> https://pytorch.org/docs/stable/onnx.html
def simple_tutorial():
	onnx_filepath = "./alexnet.onnx"

	gpu = -1
	is_cuda_available = torch.cuda.is_available()
	device = torch.device(("cuda:{}".format(gpu) if gpu >= 0 else "cuda") if is_cuda_available else "cpu")
	print("Device: {}.".format(device))

	#--------------------
	if False:
		dummy_input = torch.randn(10, 3, 224, 224, device=device)
		model = torchvision.models.alexnet(pretrained=True).to(device)

		# Providing input and output names sets the display names for values within the model's graph.
		# Setting these does not change the semantics of the graph; it is only for readability.
		#
		# The inputs to the network consist of the flat list of inputs (i.e. the values you would pass to the forward() method) followed by the flat list of parameters.
		# You can partially specify names, i.e. provide a list here shorter than the number of inputs to the model,
		# and we will only set that subset of names, starting from the beginning.
		input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
		output_names = ["output1"]

		#torch.onnx.export(model, dummy_input, onnx_filepath, verbose=True)
		torch.onnx.export(model, dummy_input, onnx_filepath, verbose=True, input_names=input_names, output_names=output_names)
		#torch.onnx.export(
		#	model, dummy_input, onnx_filepath, verbose=True, input_names=input_names, output_names=output_names,
		#	export_params=True,  # If specified, all parameters will be exported. Set this to False if you want to export an untrained model.
		#	training=torch.onnx.TrainingMode.EVAL,  # {torch.onnx.TrainingMode.EVAL, torch.onnx.TrainingMode.PRESERVE, torch.onnx.TrainingMode.TRAINING}.
		#	opset_version=9,
		#	#dynamic_axes={"actual_input_1": [0], "output1": [0]}
		#	dynamic_axes={"actual_input_1": {0: "batch"}, "output1": {0: "batch"}}
		#)
		print("ONNX model exported to {}.".format(onnx_filepath))

	#--------------------
	if True:
		# Load the ONNX model.
		model_loaded = onnx.load(onnx_filepath)
		print("ONNX model loaded from {}.".format(onnx_filepath))

		# Check that the IR is well formed.
		onnx.checker.check_model(model_loaded)

		# Print a human readable representation of the graph.
		onnx.helper.printable_graph(model_loaded.graph)

	#--------------------
	if False:
		import caffe2.python.onnx.backend as backend

		model_loaded = onnx.load(onnx_filepath)
		print("ONNX model loaded from {}.".format(onnx_filepath))

		rep = backend.prepare(model_loaded, device="CUDA:0")  # or "CPU".
		# For the Caffe2 backend:
		#	rep.predict_net is the Caffe2 protobuf for the network.
		#	rep.workspace is the Caffe2 workspace for the network.
		#		(see the class caffe2.python.onnx.backend.Workspace).
		outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))

		# To run networks with more than one input, pass a tuple rather than a single numpy ndarray.
		print(outputs[0])

	if True:
		import onnxruntime as ort

		ort_session = ort.InferenceSession(onnx_filepath)
		outputs = ort_session.run(None, {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)})

		print(outputs[0])

# REF [site] >> https://pytorch.org/docs/stable/onnx.html
#	The ONNX exporter can be both trace-based and script-based exporter.
def tracing_and_scripting_tutorial():
	onnx_filepath = "./loop.onnx"

	#--------------------
	# Trace-based only.

	class LoopModel(torch.nn.Module):
		def forward(self, x, y):
			for i in range(y):
				x = x + i
			return x

	model = LoopModel()

	dummy_input = torch.ones(2, 3, dtype=torch.long)
	loop_count = torch.tensor(5, dtype=torch.long)
	torch.onnx.export(model, (dummy_input, loop_count), onnx_filepath, verbose=True)
	print("ONNX model exported to {}.".format(onnx_filepath))

	#--------------------
	# Mixing tracing and scripting.

	@torch.jit.script
	def loop(x, y):
		for i in range(int(y)):
			x = x + i
		return x

	class LoopModel2(torch.nn.Module):
		def forward(self, x, y):
			return loop(x, y)

	model = LoopModel2()

	dummy_input = torch.ones(2, 3, dtype=torch.long)
	loop_count = torch.tensor(5, dtype=torch.long)
	torch.onnx.export(model, (dummy_input, loop_count), "./loop2.onnx", verbose=True, input_names=["input_data", "loop_range"])
	print("ONNX model exported to {}.".format("./loop2.onnx"))

	#--------------------
	# To avoid exporting a variable scalar tensor as a fixed value constant as part of the ONNX model, please avoid use of torch.Tensor.item().
	# Torch supports implicit cast of single-element tensors to numbers.

	class LoopModel(torch.nn.Module):
		def forward(self, x, y):
			res = []
			arr = x.split(2, 0)
			for i in range(int(y)):
				res += [arr[i].sum(0, False)]
			return torch.stack(res)

	model = torch.jit.script(LoopModel())

	inputs = (torch.randn(16), torch.tensor(8))
	out = model(*inputs)
	# 'example_outputs' must be provided when exporting a ScriptModule or TorchScript Function.
	torch.onnx.export(model, inputs, "./loop_and_list.onnx", opset_version=11, example_outputs=out)
	print("ONNX model exported to {}.".format("./loop_and_list.onnx"))

	#--------------------
	if False:
		import caffe2.python.onnx.backend as backend

		model_loaded = onnx.load(onnx_filepath)
		print("ONNX model loaded from {}.".format(onnx_filepath))

		rep = backend.prepare(model_loaded)
		outputs = rep.run((dummy_input.numpy(), np.array(9).astype(np.int64)))

		print(outputs[0])

	if True:
		import onnxruntime as ort

		ort_sess = ort.InferenceSession(onnx_filepath)
		outputs = ort_sess.run(None, {"input_data": dummy_input.numpy(), "loop_range": np.array(9).astype(np.int64)})

		print(outputs)

# REF [site] >> https://pytorch.org/docs/stable/onnx.html#custom-operators
def custom_operators_tutorial():
	from torch.onnx import register_custom_op_symbolic
	from torch.onnx.symbolic_helper import parse_args

	# Define custom symbolic function.
	@parse_args("v", "v", "f", "i")
	def symbolic_foo_forward(g, input1, input2, attr1, attr2):
		# TODO [check] >> What is "custom_domain::Foo"?
		return g.op("custom_domain::Foo", input1, input2, attr1_f=attr1, attr2_i=attr2)

	# Register custom symbolic function.
	register_custom_op_symbolic(symbolic_name="custom_ops::foo_forward", symbolic_fn=symbolic_foo_forward, opset_version=9)

	class FooModel(torch.nn.Module):
		def __init__(self, attr1, attr2):
			super().__init__()
			self.attr1 = attr1
			self.attr2 = attr2

		def forward(self, input1, input2):
			# Calling custom op.
			return torch.ops.custom_ops.foo_forward(input1, input2, self.attr1, self.attr2)

	model = FooModel(attr1=1.0, attr2=2)
	dummy_input1 = torch.randn(10, 3, 224, 224)
	dummy_input2 = torch.randn(10, 3, 32, 32)
	# The example above exports it as a custom operator in the "custom_domain" opset.
	torch.onnx.export(
		model,
		(dummy_input1, dummy_input2),
		"./model.onnx",
		# Only needed if you want to specify an opset version > 1.
		custom_opsets={"custom_domain": 2}  # Key(str): opset domain name, Value(int): opset version.
	)

# REF [site] >>	
#	https://onnxruntime.ai/docs/
#		ONNX Runtime for Training.
#	https://github.com/microsoft/onnxruntime-training-examples/tree/master/huggingface
#		ORTModule Examples.
def training_example():
	raise NotImplementedError

def main():
	simple_tutorial()
	#tracing_and_scripting_tutorial()

	#custom_operators_tutorial()

	#--------------------
	#training_example()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
