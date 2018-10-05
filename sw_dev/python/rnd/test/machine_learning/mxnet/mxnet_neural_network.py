import mxnet as mx
from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import datasets, transforms
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.utils import download
from mxnet import image
import numpy as np
import matplotlib.pyplot as plt
import json
from time import time

def build_network(net):
	with net.name_scope():
		net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
			nn.MaxPool2D(pool_size=2, strides=2),
			nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
			nn.MaxPool2D(pool_size=2, strides=2),
			nn.Flatten(),
			nn.Dense(120, activation='relu'),
			nn.Dense(84, activation='relu'),
			nn.Dense(10))
		return net

def build_lenet(net):
	with net.name_scope():
		net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
		net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
		net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
		net.add(gluon.nn.Flatten())
		net.add(gluon.nn.Dense(512, activation='relu'))
		net.add(gluon.nn.Dense(10))
		return net

# Create a subclass of nn.Block and implement two methods:
#	__init__() create the layers
#	forward() define the forward function.
class MixMLP(nn.Block):
	def __init__(self, **kwargs):
		# Run 'nn.Block''s init method
		super(MixMLP, self).__init__(**kwargs)
		self.blk = nn.Sequential()
		self.blk.add(nn.Dense(3, activation='relu'), nn.Dense(4, activation='relu'))
		self.dense = nn.Dense(5)

	def forward(self, x):
		y = nd.relu(self.blk(x))
		#print(y)
		return self.dense(y)

def train_model(net, trainer, loss_func, train_data, valid_data, num_epochs):
	# Create an auxiliary function to calculate the model accuracy.
	def acc(output, label):
		# output: (batch, num_output) float32 ndarray.
		# label: (batch,) int32 ndarray.
		return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

	# Train.
	for epoch in range(1, num_epochs+1):
		train_loss, train_acc, valid_acc = 0.0, 0.0, 0.0
		tic = time()
		for data, label in train_data:
			# Forward + backward.
			with autograd.record():
				output = net(data)
				loss = loss_func(output, label)
			loss.backward()

			# Update parameters.
			trainer.step(data.shape[0])

			# Calculate training metrics.
			train_loss += loss.mean().asscalar()
			train_acc += acc(output, label)

		# Calculate validation accuracy.
		for data, label in valid_data:
			valid_acc += acc(net(data), label)

		print('Epoch %d: Loss: %.3f, Train acc: %.3f, Test acc: %.3f, Time: %.1f sec' %
			(epoch, train_loss/len(train_data), train_acc/len(train_data), valid_acc/len(valid_data), time()-tic))

		"""
		# If our network is Hybrid, we can even save the network architecture into files and we won't need the network definition in a Python file to load the network.
		if isinstance(net, gluon.nn.HybridSequential):
			# export() in this case creates training_model-symbol.json and training_model-0001.params in the current directory.
			net.export('training_model', epoch=epoch)
		"""

def verify_loaded_model(net, ctx):
	def transform(data, label):
		return data.astype(np.float32)/255, label.astype(np.float32)

	# Load ten random images from the test dataset.
	sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 10, shuffle=True)

	for data, label in sample_data:
		# Display the images.
		img = nd.transpose(data, (1, 0 ,2, 3))
		img = nd.reshape(img, (28, 10*28, 1))
		imtiles = nd.tile(img, (1, 1, 3))
		plt.imshow(imtiles.asnumpy())
		plt.show()

		# Display the predictions.
		data = nd.transpose(data, (0, 3, 1, 2))
		out = net(data.as_in_context(ctx))
		predictions = nd.argmax(out, axis=1)
		print('Model predictions:', predictions.asnumpy())
		print('Ground truth:     ', label.asnumpy())

		break

# REF [site] >> https://gluon-crash-course.mxnet.io/nn.html
def create_neural_network_example():
	layer = nn.Dense(2)
	print('layer =', layer)

	# Initialize its weights with the default initialization method, which draws random values uniformly from  [?0.7,0.7].
	layer.initialize()

	# Do a forward pass with random data.
	x = nd.random.uniform(-1, 1, (3, 4))
	layer(x)

	# Can access the weight after the first forward pass.
	layer.weight.data()

	#--------------------
	# Chain layers into a neural network.
	net = build_network(nn.Sequential())
	print('net =', net)

	net.initialize()

	# Input shape is (batch_size, color_channels, height, width).
	x = nd.random.uniform(shape=(4, 1, 28, 28))
	y = net(x)

	print('y.shape =', y.shape)
	print(net[0].weight.data().shape, net[5].bias.data().shape)

	#--------------------
	net = MixMLP()
	print('net =', net)

	net.initialize()

	x = nd.random.uniform(shape=(2, 2))
	net(x)

	# Access a particular layer's weight.
	print("blk layer's weight =", net.blk[1].weight.data())

# REF [site] >> https://gluon-crash-course.mxnet.io/nn.html
def train_neural_network_example():
	mnist_train = datasets.FashionMNIST(train=True)
	X, y = mnist_train[0]
	print('X shape:', X.shape, 'X dtype:', X.dtype, 'y:', y)

	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	X, y = mnist_train[0:6]

	# Plot images.
	_, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
	for f, x, yi in zip(figs, X, y):
		# 3D -> 2D by removing the last channel dim.
		f.imshow(x.reshape((28, 28)).asnumpy())
		ax = f.axes
		ax.set_title(text_labels[int(yi)])
		ax.title.set_fontsize(20)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])
	mnist_train = mnist_train.transform_first(transformer)

	batch_size = 256
	train_data = gluon.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)

	for data, label in train_data:
		print(data.shape, label.shape)
		break

	mnist_valid = gluon.data.vision.FashionMNIST(train=False)
	valid_data = gluon.data.DataLoader(mnist_valid.transform_first(transformer), batch_size=batch_size, num_workers=4)

	# Define a model.
	net = build_network(nn.Sequential())
	net.initialize(init=init.Xavier())
	#net.collect_params().initialize(init.Xavier(), ctx=ctx)

	# Define the loss function and optimization method for training.
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
	#trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

	train_model(net, trainer, softmax_cross_entropy, train_data, valid_data, num_epochs=10)

	# Save the model parameters.
	net.save_parameters('./net.params')

# REF [site] >> https://gluon-crash-course.mxnet.io/predict.html
def predict_with_pre_trained_model_example():
	# Define a model.
	net = build_network(nn.Sequential())

	# Load the model parameters.
	net.load_parameters('./net.params')

	transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.13, 0.31)])

	# Predict the first six images in the validation dataset.
	mnist_valid = datasets.FashionMNIST(train=False)
	X, y = mnist_valid[:6]
	preds = []
	for x in X:
		x = transformer(x).expand_dims(axis=0)
		pred = net(x).argmax(axis=1)
		preds.append(pred.astype('int32').asscalar())

	# Visualize the images and compare the prediction with the ground truth.
	_, figs = plt.subplots(1, 6, figsize=(15, 15))
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	for f, x, yi, pyi in zip(figs, X, y, preds):
		f.imshow(x.reshape((28, 28)).asnumpy())
		ax = f.axes
		ax.set_title(text_labels[yi] + '\n' + text_labels[pyi])
		ax.title.set_fontsize(20)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

# REF [site] >> https://gluon-crash-course.mxnet.io/predict.html
def predict_with_models_from_gluon_model_zoo_example():
	# Gluon model zoo provides multiple pre-trained powerful models.
	#	We can download and load a pre-trained ResNet-50 V2 model that was trained on the ImageNet dataset.
	net = models.resnet50_v2(pretrained=True)

	# Download and load the text labels for each class.
	url = 'http://data.mxnet.io/models/imagenet/synset.txt'
	fname = download(url)
	with open(fname, 'r') as f:
		text_labels = [' '.join(l.split()[1:]) for l in f]

	# Randomly pick a dog image from Wikipedia as a test image, download and read it.
	url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Golden_Retriever_medium-to-light-coat.jpg/365px-Golden_Retriever_medium-to-light-coat.jpg'
	fname = download(url)
	x = image.imread(fname)

	# Use the image processing functions provided in the MXNet image module.
	x = image.resize_short(x, 256)
	x, _ = image.center_crop(x, (224, 224))
	plt.imshow(x.asnumpy())
	plt.show()

	def transform(data):
		data = data.transpose((2, 0, 1)).expand_dims(axis=0)
		rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
		rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
		return (data.astype('float32') / 255 - rgb_mean) / rgb_std

	prob = net(transform(x)).softmax()
	idx = prob.topk(k=5)[0]
	for i in idx:
		i = int(i.asscalar())
		print('With prob = %.5f, it contains %s' % (prob[0,i].asscalar(), text_labels[i]))

# REF [site] >> https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html
def save_and_load_sequential_example():
	# Use GPU if one exists, else use CPU.
	ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

	# MNIST images are 28x28. Total pixels in input layer is 28x28 = 784.
	#num_inputs = 784
	# Clasify the images into one of the 10 digits.
	#num_outputs = 10
	# 64 images in a batch.
	batch_size = 64

	# Load the training data.
	train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()), batch_size, shuffle=True)
	valid_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False).transform_first(transforms.ToTensor()), batch_size, shuffle=True)

	# Define a model.
	net = build_lenet(gluon.nn.Sequential())

	# Initialize the parameters with Xavier initializer.
	net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
	# Use cross entropy loss.
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	# Use Adam optimizer.
	trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

	train_model(net, trainer, softmax_cross_entropy, train_data, valid_data, num_epochs=10)

	#--------------------
	if True:
		model_param_filepath = './lenet.params'

		# Save model parameters to file.
		net.save_parameters(model_param_filepath)

		# Define a model.
		new_net = build_lenet(gluon.nn.Sequential())
		# Load model parameters from file.
		new_net.load_parameters(model_param_filepath, ctx=ctx)
	else:
		# NOTE [info] >> Sequential models may not be serialized as JSON files.

		model_filepath = './lenet.json'
		model_param_filepath = './lenet.params'

		# Save model architecture to file.
		sym_json = net(mx.sym.var('data')).tojson()
		sym_json = json.loads(sym_json)
		with open(model_filepath, 'w') as fd:
			json.dump(sym_json, fd, indent='\t')
		# Save model parameters to file.
		net.save_parameters(model_param_filepath)

		# Load model architecture from file.
		with open(model_filepath, 'r') as fd:
			sym_json = json.load(fd)
		sym_json = json.dumps(sym_json)
		new_net = gluon.nn.SymbolBlock(outputs=mx.sym.load_json(sym_json), inputs=mx.sym.var('data'))
		# Load model parameters from file.
		new_net.load_parameters(model_param_filepath, ctx=ctx)

	verify_loaded_model(new_net, ctx=ctx)

# REF [site] >> https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html
def save_and_load_hybrid_sequential_example():
	# Use GPU if one exists, else use CPU.
	ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

	# MNIST images are 28x28. Total pixels in input layer is 28x28 = 784.
	#num_inputs = 784
	# Clasify the images into one of the 10 digits.
	#num_outputs = 10
	# 64 images in a batch.
	batch_size = 64

	# Load the training data.
	train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()), batch_size, shuffle=True)
	valid_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False).transform_first(transforms.ToTensor()), batch_size, shuffle=True)

	# Define a model.
	net = build_lenet(gluon.nn.HybridSequential())
	net.hybridize()

	# Initialize the parameters with Xavier initializer.
	net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
	# Use cross entropy loss.
	softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
	# Use Adam optimizer.
	trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

	train_model(net, trainer, softmax_cross_entropy, train_data, valid_data, num_epochs=10)

	#--------------------
	# Save model architecture and parameters to file.
	# If our network is Hybrid, we can even save the network architecture into files and we won't need the network definition in a Python file to load the network.
	# export() in this case creates lenet_hybrid-symbol.json and lenet_hybrid-0000.params in the current directory.
	net.export('./lenet_hybrid', epoch=0)

	# Load model architecture and parameters from file.
	deserialized_net = gluon.nn.SymbolBlock.imports('./lenet_hybrid-symbol.json', ['data'], './lenet_hybrid-0000.params')

	verify_loaded_model(deserialized_net, ctx=ctx)

def main():
	#create_neural_network_example()
	#train_neural_network_example()
	#predict_with_pre_trained_model_example()
	#predict_with_models_from_gluon_model_zoo_example()

	#save_and_load_sequential_example()
	save_and_load_hybrid_sequential_example()  # NOTE [info] >> Better choice.

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
