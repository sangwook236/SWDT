from mxnet import nd, gpu, gluon, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from time import time

# REF [site] >>
#	https://gluon-crash-course.mxnet.io/use_gpus.html
#	https://gluon-crash-course.mxnet.io/mxnet_packages.html
def basic_example():
	# Allocate data to a GPU.
	# MXNet's NDArray is very similar to Numpy.
	# One major difference is NDArray has a context attribute that specifies which device this array is on.
	# By default, it is cpu().
	x = nd.ones((3, 4), ctx=gpu())

	# Copy x to the second GPU, gpu(1).
	x.copyto(gpu(1))

	# Run an operation on a GPU.
	y = nd.random.uniform(shape=(3, 4), ctx=gpu())
	print('x + y =', x + y)

	# Run a neural network on a GPU.
	net = nn.Sequential()
	net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
		nn.MaxPool2D(pool_size=2, strides=2),
		nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
		nn.MaxPool2D(pool_size=2, strides=2),
		nn.Flatten(),
		nn.Dense(120, activation='relu'),
		nn.Dense(84, activation='relu'),
		nn.Dense(10))

	# Load the saved parameters into GPU 0 directly, or use net.collect_params().reset_ctx to change the device.
	net.load_parameters('./net.params', ctx=gpu(0))

	# Create input data on GPU 0.
	x = nd.random.uniform(shape=(1,1,28,28), ctx=gpu(0))

	# The forward function will then run on GPU 0.
	print('net(x) =', net(x))

	# Multi-GPU training.
	batch_size = 256
	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
	train_data = gluon.data.DataLoader(datasets.FashionMNIST(train=True).transform_first(transformer), batch_size, shuffle=True, num_workers=4)
	valid_data = gluon.data.DataLoader(datasets.FashionMNIST(train=False).transform_first(transformer), batch_size, shuffle=False, num_workers=4)

	# Diff 1: Use two GPUs for training.
	devices = [gpu(0), gpu(1)]

	# Diff 2: reinitialize the parameters and place them on multiple GPUs.
	net.collect_params().initialize(force_reinit=True, ctx=devices)

	# Loss and trainer are the same as before.
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

	for epoch in range(10):
		train_loss = 0.
		tic = time()
		for data, label in train_data:
			# Diff 3: split batch and load into corresponding devices.
			data_list = gluon.utils.split_and_load(data, devices)
			label_list = gluon.utils.split_and_load(label, devices)

			# Diff 4: run forward and backward on each devices.
			# MXNet will automatically run them in parallel.
			with autograd.record():
				losses = [softmax_cross_entropy(net(X), y) for X, y in zip(data_list, label_list)]
			for l in losses:
				l.backward()

			trainer.step(batch_size)

			# Diff 5: sum losses over all devices.
			train_loss += sum([l.sum().asscalar() for l in losses])

		print('Epoch %d: Loss: %.3f, Time %.1f sec' % (epoch, train_loss/len(train_data)/batch_size, time()-tic))

def main():
	basic_example()

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
