#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import torchvision

# REF [site] >> https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
def profiler_recipe_tutorial():
	# Instantiate a simple ResNet model
	model = torchvision.models.resnet18()
	inputs = torch.randn(5, 3, 224, 224)

	# Using profiler to analyze execution time.
	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
		with torch.profiler.record_function("model_inference"):
			model(inputs)

	# Let's print out the stats for the execution above
	print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

	# To get a finer granularity of results and include operator input shapes, pass group_by_input_shape=True (note: this requires running the profiler with record_shapes=True)
	print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

	#-----
	# Profiler can also be used to analyze performance of models executed on GPUs
	model = torchvision.models.resnet18().cuda()
	inputs = torch.randn(5, 3, 224, 224).cuda()

	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
		with torch.profiler.record_function("model_inference"):
			model(inputs)

	print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

	#-----
	# Using profiler to analyze memory consumption
	#	PyTorch profiler can also show the amount of memory (used by the model's tensors) that was allocated (or released) during the execution of the model's operators
	#	To enable memory profiling functionality pass profile_memory=True
	model = torchvision.models.resnet18()
	inputs = torch.randn(5, 3, 224, 224)

	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
		model(inputs)

	print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
	print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

	#-----
	# Using tracing functionality
	#	Profiling results can be outputted as a .json trace file
	model = torchvision.models.resnet18().cuda()
	inputs = torch.randn(5, 3, 224, 224).cuda()

	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
		model(inputs)

	prof.export_chrome_trace("./trace.json")

	#-----
	# Examining stack traces
	#	Profiler can be used to analyze Python and TorchScript stack traces
	with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], with_stack=True) as prof:
		model(inputs)

	# Print aggregated stats
	print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

	#-----
	# Visualizing data as a flame graph
	#	Execution time (self_cpu_time_total and self_cuda_time_total metrics) and stack traces can also be visualized as a flame graph
	prof.export_stacks("./profiler_stacks.txt", "self_cuda_time_total")

	# Recommend using Flamegraph tool to generate an interactive .svg file
	"""
	git clone https://github.com/brendangregg/FlameGraph
	cd FlameGraph
	./flamegraph.pl --title "CUDA time" --countname "us." /tmp/profiler_stacks.txt > perf_viz.svg
	"""

	#-----
	# Using profiler to analyze long-running jobs
	my_schedule = torch.profiler.schedule(
		skip_first=10,
		wait=5,
		warmup=1,
		active=3,
		repeat=2,
	)

	def trace_handler(p):
		output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
		print(output)
		p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

	with profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
		on_trace_ready=trace_handler
	) as p:
		for idx in range(8):
			model(inputs)
			p.step()

# REF [site] >>
#	https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
#	https://github.com/pytorch/kineto/tree/main/tb_plugin
def tensorboard_profiler_tutorial():

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	# Prepare the input data
	transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(224),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])
	train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

	# Create Resnet model, loss function, and optimizer objects
	model = torchvision.models.resnet18(weights="IMAGENET1K_V1").cuda(device)
	criterion = torch.nn.CrossEntropyLoss().cuda(device)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	model.train()

	# Define the training step for each batch of input data
	def train(data):
		inputs, labels = data[0].to(device=device), data[1].to(device=device)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Use profiler to record execution events
	if True:
		# The profiler is enabled through the context manager and accepts several parameters
		with torch.profiler.profile(
			schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
			on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/resnet18"),
			record_shapes=True,
			profile_memory=True,
			with_stack=True
		) as prof:
			for step, batch_data in enumerate(train_loader):
				prof.step()  # Need to call this at each step to notify profiler of steps" boundary
				if step >= 1 + 1 + 3:
					break
				train(batch_data)
	else:
		# The following non-context manager start/stop is supported as well
		prof = torch.profiler.profile(
			schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
			on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/resnet18"),
			record_shapes=True,
			with_stack=True
		)
		prof.start()
		for step, batch_data in enumerate(train_loader):
			prof.step()
			if step >= 1 + 1 + 3:
				break
			train(batch_data)
		prof.stop()

	# Run the profiler
	#	Run the above code. The profiling result will be saved under ./log/resnet18 directory

	# Use TensorBoard to view results and analyze model performance
	#	Install PyTorch Profiler TensorBoard Plugin:
	#		pip install torch-tb-profiler
	#	Launch the TensorBoard:
	#		tensorboard --logdir=./log/resnet18
	#	Open the TensorBoard profile URL:
	#		http://localhost:6006/#pytorch_profiler

def main():
	# https://pytorch.org/docs/master/profiler.html

	profiler_recipe_tutorial()

	tensorboard_profiler_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
