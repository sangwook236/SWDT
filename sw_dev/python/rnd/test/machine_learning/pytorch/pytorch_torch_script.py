#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch, torchvision

# REF [site] >> https://pytorch.org/tutorials/advanced/cpp_export.html
def simple_tutorial():
	# Convert a PyTorch model to Torch Script.
	if True:
		# Convert to Torch Script via tracing.

		torch_script_filepath = "./resnet_ts_model.pth"

		# An instance of your model.
		model = torchvision.models.resnet18()

		# An example input you would normally provide to your model's forward() method.
		example = torch.rand(1, 3, 224, 224)

		# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
		script_module = torch.jit.trace(model, example)

		output = script_module(torch.ones(1, 3, 224, 224))
		print(output[0, :5])
	else:
		# Convert to Torch Script via annotation.

		torch_script_filepath = "./lenet_mnist_ts_model.pth"

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
		script_module = torch.jit.script(my_module)

		output = script_module(torch.rand(1, 1, 28, 28))
		#output = script_module(torch.rand(1, 1, 32, 32))
		print(output)

	# Serialize a script module to a file.
	script_module.save(torch_script_filepath)

	# Load and execute the script module in C++.
	#	REF [file] >>
	#		${SWDT_CPP_HOME}/rnd/test/machine_learning/torch/torch_torch_script_example.cpp
	#		${SWDT_CPP_HOME}/rnd/test/machine_learning/torch/torch_training_example.cpp

def main():
	simple_tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
