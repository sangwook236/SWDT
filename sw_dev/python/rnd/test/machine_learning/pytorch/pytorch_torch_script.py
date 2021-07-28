#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import torch, torchvision

# REF [site] >> https://pytorch.org/tutorials/advanced/cpp_export.html
def simple_tutorial():
	# Convert a PyTorch model to Torch Script.
	if True:
		# Convert to Torch Script via tracing.

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

		class MyModule(torch.nn.Module):
			def __init__(self, N, M):
				super(MyModule, self).__init__()
				self.weight = torch.nn.Parameter(torch.rand(N, M))

			def forward(self, input):
				if input.sum() > 0:
					output = self.weight.mv(input)
				else:
					output = self.weight + input
				return output

		my_module = MyModule(10, 20)
		script_module = torch.jit.script(my_module)

		output = script_module(torch.ones(20))
		print(output)

	# Serialize a script module to a file.
	script_module.save("./resnet_ts_model.pt")

	# Load the script module in C++.
	#	REF [cpp] >> ${SWDT_CPP_HOME}/rnd/test/machine_learning/torch/torch_torch_script_example.cpp

def main():
	simple_tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
