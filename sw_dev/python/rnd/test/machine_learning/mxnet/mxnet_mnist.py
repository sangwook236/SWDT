# REF [site] >> https://mxnet.incubator.apache.org/tutorials/gluon/mnist.html

from __future__ import print_function
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as nd

def mnist_mlp():
	# Fix the random seed.
	mx.random.seed(42)

	mnist = mx.test_utils.get_mnist()

	batch_size = 100
	train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
	val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

	# Define a network.
	net = nn.Sequential()
	with net.name_scope():
		net.add(nn.Dense(128, activation='relu'))
		net.add(nn.Dense(64, activation='relu'))
		net.add(nn.Dense(10))

	# Set the context on GPU is available otherwise CPU.
	gpus = mx.test_utils.list_gpus()
	ctx = [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
	net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.02})

	# Train the network.
	epoch = 10
	# Use Accuracy as the evaluation metric.
	metric = mx.metric.Accuracy()
	softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

	for i in range(epoch):
		# Reset the train data iterator.
		train_data.reset()
		# Loop over the train data iterator.
		for batch in train_data:
			# Splits train data into multiple slices along batch_axis and copy each slice into a context.
			data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
			# Splits train labels into multiple slices along batch_axis and copy each slice into a context.
			label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
			outputs = []
			# Inside training scope.
			with ag.record():
				for x, y in zip(data, label):
					z = net(x)
					# Computes softmax cross entropy loss.
					loss = softmax_cross_entropy_loss(z, y)
					# Backpropagate the error for one iteration.
					loss.backward()
					outputs.append(z)
			# Updates internal evaluation.
			metric.update(label, outputs)
			# Make one step of parameter update.
			#	Trainer needs to know the batch size of data to normalize the gradient by 1/batch_size.
			trainer.step(batch.data[0].shape[0])

		# Gets the evaluation result.
		name, acc = metric.get()
		# Reset evaluation result to initial state.
		metric.reset()
		print('Training acc at epoch %d: %s = %f' % (i, name, acc))

	# Predict.
	# Use accuracy as the evaluation metric.
	metric = mx.metric.Accuracy()

	# Reset the validation data iterator.
	val_data.reset()
	# Loop over the validation data iterator.
	for batch in val_data:
		# Splits validation data into multiple slices along batch_axis and copy each slice into a context.
		data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
		# Splits validation label into multiple slices along batch_axis and copy each slice into a context.
		label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
		outputs = []
		for x in data:
			outputs.append(net(x))
		# Updates internal evaluation.
		metric.update(label, outputs)

	print('Validation acc: %s = %f' % metric.get())
	assert metric.get()[1] > 0.94

class Net(gluon.Block):
	def __init__(self, **kwargs):
		super(Net, self).__init__(**kwargs)
		with self.name_scope():
			# Layers created in name_scope will inherit name space from parent layer.
			self.conv1 = nn.Conv2D(20, kernel_size=(5, 5))
			self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
			self.conv2 = nn.Conv2D(50, kernel_size=(5, 5))
			self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
			self.fc1 = nn.Dense(500)
			self.fc2 = nn.Dense(10)

	def forward(self, x):
		x = self.pool1(nd.tanh(self.conv1(x)))
		x = self.pool2(nd.tanh(self.conv2(x)))
		# 0 means copy over size from corresponding dimension.
		# -1 means infer size from the rest of dimensions.
		x = x.reshape((0, -1))
		x = nd.tanh(self.fc1(x))
		x = nd.tanh(self.fc2(x))
		return x

def mnist_cnn():
	# Fix the random seed.
	mx.random.seed(42)

	mnist = mx.test_utils.get_mnist()

	batch_size = 100
	train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
	val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

	# Create a network.
	net = Net()

	# Set the context on GPU is available otherwise CPU.
	ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
	net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

	# Train the network.
	epoch = 10
	# Use accuracy as the evaluation metric.
	metric = mx.metric.Accuracy()
	softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()

	for i in range(epoch):
		# Reset the train data iterator.
		train_data.reset()
		# Loop over the train data iterator.
		for batch in train_data:
			# Splits train data into multiple slices along batch_axis and copy each slice into a context.
			data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
			# Splits train labels into multiple slices along batch_axis and copy each slice into a context.
			label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
			outputs = []
			# Inside training scope.
			with ag.record():
				for x, y in zip(data, label):
					z = net(x)
					# Computes softmax cross entropy loss.
					loss = softmax_cross_entropy_loss(z, y)
					# Backpropogate the error for one iteration.
					loss.backward()
					outputs.append(z)
			# Updates internal evaluation.
			metric.update(label, outputs)
			# Make one step of parameter update.
			#	Trainer needs to know the batch size of data to normalize the gradient by 1/batch_size.
			trainer.step(batch.data[0].shape[0])

		# Gets the evaluation result.
		name, acc = metric.get()
		# Reset evaluation result to initial state.
		metric.reset()
		print('Training acc at epoch %d: %s = %f' % (i, name, acc))

	# Predict.
	# Use accuracy as the evaluation metric.
	metric = mx.metric.Accuracy()

	# Reset the validation data iterator.
	val_data.reset()
	# Loop over the validation data iterator.
	for batch in val_data:
		# Splits validation data into multiple slices along batch_axis and copy each slice into a context.
		data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
		# Splits validation label into multiple slices along batch_axis and copy each slice into a context.
		label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
		outputs = []
		for x in data:
			outputs.append(net(x))
		# Updates internal evaluation.
		metric.update(label, outputs)

	print('Validation acc: %s = %f' % metric.get())
	assert metric.get()[1] > 0.98

def main():
	mnist_mlp()
	mnist_cnn()

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
