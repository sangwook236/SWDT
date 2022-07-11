#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# NOTE [info] >>
#	https://pytorch.org/docs/stable/jit.html
#	https://pytorch.org/docs/stable/jit_language_reference.html
#	https://pytorch.org/docs/stable/jit_builtin_functions.html
#	https://pytorch.org/docs/stable/jit_unsupported.html
#
#	TorchScript is a statically typed subset of Python that can either be written directly (using the @torch.jit.script decorator) or generated automatically from Python code via tracing.
#	When using tracing, code is automatically converted into this subset of Python by recording only the actual operators on tensors and simply executing and discarding the other surrounding Python code.
#	When writing TorchScript directly using @torch.jit.script decorator, the programmer must only use the subset of Python supported in TorchScript.
#
#	In many cases either tracing or scripting is an easier approach for converting a model to TorchScript.
#	Tracing and scripting can be composed to suit the particular requirements of a part of a model.
#
#	As a subset of Python, any valid TorchScript function is also a valid Python function.
#	Unlike Python, each variable in TorchScript function must have a single static type.
#	TorchScript does not support all features and types of the typing module.

# NOTE [info] {important} >>
#	All parameters in ScriptModule and ScriptFunction are regarded as torch.Tensor.

import torch, torchvision
#from torch.hub import VAR_DEPENDENCY

# REF [site] >> https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
def beginner_tutorial():
	gpu = -1
	is_cuda_available = torch.cuda.is_available()
	device = torch.device(('cuda:{}'.format(gpu) if gpu >= 0 else 'cuda') if is_cuda_available else 'cpu')
	print('Device: {}.'.format(device))

	x = torch.rand(3, 4, device=device)
	h = torch.rand(3, 4, device=device)
	xs = torch.rand(10, 3, 4, device=device)

	class MyDecisionGate(torch.nn.Module):
		def forward(self, x):
			# NOTE [warning] >> TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
			if x.sum() > 0:
				return x
			else:
				return -x

	class MyCell(torch.nn.Module):
		def __init__(self):
			super(MyCell, self).__init__()
			# NOTE [info] >> It doesn't matter which one is used.
			self.dg = MyDecisionGate()
			#self.dg = MyDecisionGate().to(device)
			self.linear = torch.nn.Linear(4, 4)

		def forward(self, x, h):
			new_h = torch.tanh(self.dg(self.linear(x)) + h)
			return new_h, new_h

	my_cell = MyCell().to(device)

	if False:
		print("my_cell(x, h) = {}.".format(my_cell(x, h)))
		print("---------- Model (my_cell):\n{}.".format(my_cell))

	#--------------------
	# Basics of TorchScript.

	# Tracing modules.
	traced_cell = torch.jit.trace(my_cell, (x, h))

	if False:
		print("traced_cell(x, h) = {}.".format(traced_cell(x, h)))
		print("---------- Model (traced_cell):\n{}.".format(traced_cell))
		print("---------- Graph (traced_cell):\n{}.".format(traced_cell.graph))
		print("---------- Code (traced_cell):\n{}.".format(traced_cell.code))
		print("---------- Model (traced_cell.dg):\n{}.".format(traced_cell.dg))
		print("---------- Graph (traced_cell.dg):\n{}.".format(traced_cell.dg.graph))
		print("---------- Code (traced_cell.dg):\n{}.".format(traced_cell.dg.code))

	# Using scripting to convert modules.
	class MyCell2(torch.nn.Module):
		def __init__(self, dg):
			super(MyCell2, self).__init__()
			self.dg = dg
			self.linear = torch.nn.Linear(4, 4)

		def forward(self, x, h):
			new_h = torch.tanh(self.dg(self.linear(x)) + h)
			return new_h, new_h

	my_cell2 = MyCell2(MyDecisionGate()).to(device)
	traced_cell2 = torch.jit.trace(my_cell2, (x, h))

	if True:
		print("traced_cell2(x, h) = {}.".format(traced_cell2(x, h)))
		print("---------- Graph (traced_cell2):\n{}.".format(traced_cell2.graph))
		print("---------- Code (traced_cell2):\n{}.".format(traced_cell2.code))
		print("---------- Graph (traced_cell2.dg):\n{}.".format(traced_cell2.dg.graph))
		print("---------- Code (traced_cell2.dg):\n{}.".format(traced_cell2.dg.code))

	# NOTE [info] >> It doesn't matter which one is used.
	scripted_gate = torch.jit.script(MyDecisionGate())
	#scripted_gate = torch.jit.script(MyDecisionGate().to(device))

	my_cell2 = MyCell2(scripted_gate).to(device)
	scripted_cell2 = torch.jit.script(my_cell2)

	if True:
		print("scripted_cell2(x, h) = {}.".format(scripted_cell2(x, h)))
		print("---------- Graph (scripted_cell2):\n{}.".format(scripted_cell2.graph))
		print("---------- Code (scripted_cell2):\n{}.".format(scripted_cell2.code))
		print("---------- Graph (scripted_gate):\n{}.".format(scripted_gate.graph))
		print("---------- Code (scripted_gate):\n{}.".format(scripted_gate.code))
		print("---------- Graph (scripted_cell2.dg):\n{}.".format(scripted_cell2.dg.graph))
		print("---------- Code (scripted_cell2.dg):\n{}.".format(scripted_cell2.dg.code))

	# Mixing scripting and tracing.
	#@torch.jit.script  # RuntimeError: Type '<class '__main__.beginner_tutorial.<locals>.__init__'>' cannot be compiled since it inherits from nn.Module, pass an instance instead.
	class MyRNNLoop(torch.nn.Module):
		def __init__(self, device):
			super(MyRNNLoop, self).__init__()
			self.device = device
			# NOTE [error] >> RuntimeError: Tensor for 'out' is on CPU, Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm).
			#	Device conversion is required.
			self.cell = torch.jit.trace(MyCell2(scripted_gate).to(self.device), (x, h))

		def forward(self, xs):
			# NOTE [error] >> python value of type 'device' cannot be used as a value. Perhaps it is a closed over global variable? If so, please consider passing it in as an argument or use a local varible instead.
			#	A member variable, self.device is added.
			h, y = torch.zeros(3, 4, device=self.device), torch.zeros(3, 4, device=self.device)
			for i in range(xs.size(0)):
				y, h = self.cell(xs[i], h)
			return y, h

	# NOTE [info] >> It doesn't matter which one is used.
	scripted_rnn_loop = torch.jit.script(MyRNNLoop(device))
	#scripted_rnn_loop = torch.jit.script(MyRNNLoop(device).to(device))

	if True:
		#print("The type of scripted_rnn_loop = {}.".format(type(scripted_rnn_loop)))  # torch.jit.RecursiveScriptModule.
		print("scripted_rnn_loop(xs) = {}.".format(scripted_rnn_loop(xs)))
		print("---------- Graph (scripted_rnn_loop):\n{}.".format(scripted_rnn_loop.graph))
		print("---------- Code (scripted_rnn_loop):\n{}.".format(scripted_rnn_loop.code))
		print("---------- Graph (scripted_rnn_loop.cell):\n{}.".format(scripted_rnn_loop.cell.graph))
		print("---------- Code (scripted_rnn_loop.cell):\n{}.".format(scripted_rnn_loop.cell.code))

	class WrapRNN(torch.nn.Module):
		def __init__(self):
			super(WrapRNN, self).__init__()
			self.loop = torch.jit.script(MyRNNLoop(device).to(device))  # NOTE [info] >> It gives better results.
			# NOTE [warning] >> TracerWarning: Converting a tensor to a Python index might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
			#self.loop = MyRNNLoop(device).to(device)

		def forward(self, xs):
			y, h = self.loop(xs)
			return torch.relu(y)

	traced_wrap_rnn = torch.jit.trace(WrapRNN().to(device), (xs))

	if True:
		#print("The type of traced_wrap_rnn = {}.".format(type(traced_wrap_rnn)))  # torch.jit.TopLevelTracedModule.
		print("traced_wrap_rnn(xs) = {}.".format(traced_wrap_rnn(xs)))
		print("---------- Code (traced_wrap_rnn):\n{}.".format(traced_wrap_rnn.code))
		print("---------- Graph (traced_wrap_rnn):\n{}.".format(traced_wrap_rnn.graph))

	#--------------------
	# Saving and loading models.

	torch_script_filepath = "./wrapped_rnn_ts.pth"
	traced_wrap_rnn.save(torch_script_filepath)
	print("Wrapped RNN model saved to {}.".format(torch_script_filepath))

	traced_wrap_rnn_loaded = torch.jit.load(torch_script_filepath)
	print("Wrapped RNN model loaded from {}.".format(torch_script_filepath))

	if False:
		print("traced_wrap_rnn_loaded((torch.rand(10, 3, 4))) = {}.".format(traced_wrap_rnn_loaded((torch.rand(10, 3, 4)))))
		print("---------- Model (traced_wrap_rnn_loaded):\n{}.".format(traced_wrap_rnn_loaded))
		print("---------- Graph (traced_wrap_rnn_loaded):\n{}.".format(traced_wrap_rnn_loaded.graph))
		print("---------- Code (traced_wrap_rnn_loaded):\n{}.".format(traced_wrap_rnn_loaded.code))
		print("---------- Model (traced_wrap_rnn_loaded.loop):\n{}.".format(traced_wrap_rnn_loaded.loop))
		print("---------- Graph (traced_wrap_rnn_loaded.loop):\n{}.".format(traced_wrap_rnn_loaded.loop.graph))
		print("---------- Code (traced_wrap_rnn_loaded.loop):\n{}.".format(traced_wrap_rnn_loaded.loop.code))

# REF [site] >> https://pytorch.org/docs/stable/jit.html
def simple_tutorial():
	# Mixing tracing and scripting.

	# Scripted functions can call traced functions.
	def foo(x, y):
		return 2 * x + y

	traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

	@torch.jit.script
	def bar(x):
		return traced_foo(x, x)

	#print("The type of scripted function = {}.".format(type(bar)))  # torch.jit.ScriptFunction.

	torch_script_filepath = "./scripted_func_ts.pth"
	bar.save(torch_script_filepath)
	print("Scripted function saved to {}.".format(torch_script_filepath))

	#--------------------
	# Traced functions can call script functions.
	@torch.jit.script
	def foo(x, y):
		if x.max() > y.max():
			r = x
		else:
			r = y
		return r

	def bar(x, y, z):
		return foo(x, y) + z

	traced_bar = torch.jit.trace(bar, (torch.rand(3), torch.rand(3), torch.rand(3)))

	#print("The type of traced function = {}.".format(type(traced_bar)))  # torch.jit.ScriptFunction.

	torch_script_filepath = "./traced_func_ts.pth"
	traced_bar.save(torch_script_filepath)
	print("Traced function saved to {}.".format(torch_script_filepath))

	#--------------------
	# This composition also works for nn.Modules as well.
	class MyScriptModule(torch.nn.Module):
		def __init__(self):
			super(MyScriptModule, self).__init__()
			self.means = torch.nn.Parameter(torch.tensor([103.939, 116.779, 123.68]).resize_(1, 3, 1, 1))
			self.resnet = torch.jit.trace(torchvision.models.resnet18(), torch.rand(1, 3, 224, 224))

		def forward(self, input):
			return self.resnet(input - self.means)

	my_script_module = torch.jit.script(MyScriptModule())

	#print("The type of script module = {}.".format(type(my_script_module)))  # torch.jit.RecursiveScriptModule.

	torch_script_filepath = "./module_ts.pth"
	my_script_module.save(torch_script_filepath)
	print("Script module saved to {}.".format(torch_script_filepath))

# REF [site] >> https://pytorch.org/docs/stable/jit.html
#	Setting the environment variable PYTORCH_JIT=0 will disable all script and tracing annotations.
#	If there is hard-to-debug error in one of your TorchScript models, you can use this flag to force everything to run using native Python.
#	Since TorchScript (scripting and tracing) is disabled with this flag, you can use tools like pdb to debug the model code.
def debugging_tutorial():
	@torch.jit.script
	def scripted_fn(x : torch.Tensor):
		for i in range(12):
			x = x + x
		return x

	def fn(x):
		x = torch.neg(x)
		import pdb; pdb.set_trace()
		return scripted_fn(x)

	traced_fn = torch.jit.trace(fn, (torch.rand(4, 5),))
	traced_fn(torch.rand(3, 4))

# REF [site] >> https://pytorch.org/docs/stable/jit.html
def inspecting_code_tutorial():
	@torch.jit.script
	def foo(len):
		# type: (int) -> torch.Tensor
		rv = torch.zeros(3, 4)
		for i in range(len):
			if i < 10:
				rv = rv - 1.0
			else:
				rv = rv + 1.0
		return rv

	# A ScriptModule with a single forward method will have an attribute code, which you can use to inspect the ScriptModule's code.
	# If the ScriptModule has more than one method, you will need to access .code on the method itself and not the module.
	print(foo.code)

# REF [site] >> https://pytorch.org/docs/stable/jit.html
def interpreting_graphs_tutorial():
	@torch.jit.script
	def foo(len):
		# type: (int) -> torch.Tensor
		rv = torch.zeros(3, 4)
		for i in range(len):
			if i < 10:
				rv = rv - 1.0
			else:
				rv = rv + 1.0
		return rv

	# TorchScript IR Graph.
	# TorchScript uses a static single assignment (SSA) intermediate representation (IR) to represent computation.
	# The instructions in this format consist of ATen (the C++ backend of PyTorch) operators and other primitive operators, including control flow operators for loops and conditionals.
	print(foo.graph)

# REF [site] >> https://pytorch.org/docs/stable/jit.html
def tracer_tutorial():
	# Automatic trace checking.

	if False:
		def loop_in_traced_fn(x):
			result = x[0]
			for i in range(x.size(0)):
				result = result * x[i]
			return result

		inputs = (torch.rand(3, 4, 5),)
		check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

		# One way to automatically catch many errors in traces is by using check_inputs on the torch.jit.trace() API. 
		traced = torch.jit.trace(loop_in_traced_fn, inputs, check_inputs=check_inputs)

	if False:
		def fn(x):
			result = x[0]
			for i in range(x.size(0)):
				result = result * x[i]
			return result

		inputs = (torch.rand(3, 4, 5),)
		check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

		scripted_fn = torch.jit.script(fn)
		print(scripted_fn.graph)
		#print(str(scripted_fn.graph).strip())

		# Data-dependent control flow can be captured using torch.jit.script().
		for input_tuple in [inputs] + check_inputs:
			torch.testing.assert_allclose(fn(*input_tuple), scripted_fn(*input_tuple))

	#--------------------
	# Tracer warnings.

	if True:
		def fill_row_zero(x):
			#x[0] = torch.rand(*x.shape[1:2])  # An in-place assignment on a slice (a view) of a Tensor.
			x = torch.cat((torch.rand(1, *x.shape[1:2]), x[1:2]), dim=0)  # The code to not use the in-place update. The result tensor out-of-place with torch.cat.
			return x

		traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
		print(traced.graph)

# REF [site] >> https://pytorch.org/tutorials/advanced/cpp_export.html
def cpp_export_tutorial():
	gpu = -1
	is_cuda_available = torch.cuda.is_available()
	device = torch.device(('cuda:{}'.format(gpu) if gpu >= 0 else 'cuda') if is_cuda_available else 'cpu')
	print('Device: {}.'.format(device))

	# Convert a PyTorch model to Torch Script.
	if True:
		# Convert to Torch Script via tracing.

		# NOTE [info] >> https://pytorch.org/docs/stable/generated/torch.jit.trace.html
		#	Tracing is ideal for code that operates only on Tensors and lists, dictionaries, and tuples of Tensors.
		#	Tracing only correctly records functions and modules which are not data dependent (e.g., do not have conditionals on data in tensors) and do not have any untracked external dependencies (e.g., perform input/output or access global variables).
		#	Tracing only records operations done when the given function is run on the given tensors. Therefore, the returned ScriptModule will always run the same traced graph on any input.

		torch_script_filepath = "./resnet_ts_model.pth"
		input_shape = 1, 3, 224, 224

		# An instance of your model.
		model = torchvision.models.resnet18()
		if is_cuda_available:
			model = model.to(device)

		# An example input you would normally provide to your model's forward() method.
		# NOTE [info] >> The shape of a dummy input is fixed in Torch Script. (?)
		dummy_input = torch.rand(*input_shape)
		if is_cuda_available:
			dummy_input = dummy_input.to(device)

		# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
		script_module = torch.jit.trace(model, dummy_input)  # torch.jit.TopLevelTracedModule.
		# NOTE [info] >> This is not working.
		#if is_cuda_available:
		#	script_module = script_module.to(device)

		script_module.eval()
		if is_cuda_available:
			output = script_module(torch.ones(*input_shape).to(device)).cpu()
		else:
			output = script_module(torch.ones(*input_shape))
		print('Output: {}.'.format(output[0, :5]))
	else:
		# Convert to Torch Script via annotation.

		torch_script_filepath = "./lenet_mnist_ts_model.pth"
		input_shape = 1, 1, 28, 28  # For 28x28 input.
		#input_shape = 1, 1, 32, 32  # For 32x32 input.

		# REF [file] >> ./pytorch_neural_network.py
		class MyModule(torch.nn.Module):
			def __init__(self):
				super(MyModule, self).__init__()

				# 1 input image channel, 6 output channels, 3x3 square convolution kernel.
				self.conv1 = torch.nn.Conv2d(1, 6, 3)
				self.conv2 = torch.nn.Conv2d(6, 16, 3)
				# An affine operation: y = Wx + b.
				self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # For 28x28 input.
				#self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # For 32x32 input.
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

		my_module = MyModule()
		if is_cuda_available:
			my_module = my_module.to(device)

		script_module = torch.jit.script(my_module)  # torch.jit.RecursiveScriptModule.
		# NOTE [info] >> This is working.
		#if is_cuda_available:
		#	script_module = script_module.to(device)

		script_module.eval()
		if is_cuda_available:
			output = script_module(torch.rand(*input_shape).to(device)).cpu()
		else:
			output = script_module(torch.rand(*input_shape))
		print('Output: {}.'.format(output))

	# Serialize a script module to a file.
	script_module.save(torch_script_filepath)

	# Load and execute the script module in C++.
	#	REF [file] >>
	#		${SWDT_CPP_HOME}/rnd/test/machine_learning/torch/torch_torch_script_example.cpp
	#		${SWDT_CPP_HOME}/rnd/test/machine_learning/torch/torch_training_example.cpp

def main():
	# REF [file] >> ./pytorch_tensorrt.py

	beginner_tutorial()
	#simple_tutorial()

	#debugging_tutorial()  # Usage: PYTORCH_JIT=0 python pytorch_torch_script.py
	#inspecting_code_tutorial()
	#interpreting_graphs_tutorial()
	#tracer_tutorial()

	#cpp_export_tutorial()

	#--------------------
	# Triton Inference Server + TorchScript.
	#   Refer to triton_usage_guide.txt

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
