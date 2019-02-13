#!/usr/bin/env python

# export PYTHONPATH=${CAFFE_HOME}/python:$PYTHONPATH
# export LD_LIBRARY_PATH=${CAFFE_HOME}/build/install/lib:$LD_LIBRARY_PATH

import os, time
import numpy as np
import pandas as pd
import caffe
from caffe.proto import caffe_pb2
from PIL import Image
from google.protobuf import text_format

# Download the latest caffe.proto.
# Compile into python library.
#	protoc --python_out=. caffe.proto
#		Generates caffe_pb2.py.
#		caffe_pb2.py also exists in ${CAFFE_HOME}/python/caffe/proto after making.
# Import and parse.
#	import caffe_pb2
def parse_caffe_model():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	# Trained Caffe model file.
	caffe_model_filepath = caffe_home_dir_path + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
	#caffe_model_filepath = caffe_home_dir_path + '/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	#caffe_model_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	#caffe_model_filepath = './yolov3.caffemodel'

	# REF [site] >> https://www.programcreek.com/python/example/104218/caffe.proto.caffe_pb2.NetParameter
	netParam  = caffe_pb2.NetParameter()
	with open(caffe_model_filepath, 'rb') as fd:
		netParam.ParseFromString(fd.read())

	#print(netParam)
	print('#layers =', len(netParam.layer))

	for layer in netParam.layer:
		shapes = list()
		for blob in layer.blobs:
			weights = np.reshape(np.array(blob.data), blob.shape.dim)
			shapes.append(weights.shape)
		print('{}: {}'.format(layer.name, shapes))

def parse_caffe_prototxt():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	# Network definition file.
	#deploy_prototxt_filepath = caffe_home_dir_path + '/models/bvlc_alexnet/deploy.prototxt'
	#deploy_prototxt_filepath = caffe_home_dir_path + '/models/bvlc_googlenet/deploy.prototxt'
	deploy_prototxt_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/deploy.prototxt'

	# REF [site] >> https://www.programcreek.com/python/example/104218/caffe.proto.caffe_pb2.NetParameter
	netParam  = caffe_pb2.NetParameter()
	with open(deploy_prototxt_filepath, 'r') as fd:
		text_format.Merge(fd.read(), netParam)

	#print(netParam)
	print('#layers =', len(netParam.layer))

def numpy_to_blobproto():
	arr = np.random.rand(2, 3, 4)
	print(arr)

	blob = caffe.io.array_to_blobproto(arr)
	print(blob)

	arr2 = caffe.io.blobproto_to_array(blob)
	print(arr2)

def load_npy_weights_into_caffe_model():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	# Network definition file.
	#prototxt_filepath = caffe_home_dir_path + '/models/bvlc_alexnet/deploy.prototxt'
	#prototxt_filepath = caffe_home_dir_path + '/models/bvlc_googlenet/deploy.prototxt'
	#prototxt_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/deploy.prototxt'
	prototxt_filepath = './yolov3.prototxt'

	#--------------------
	net = caffe.Net(prototxt_filepath, caffe.TEST)

	print('#layers =', len(net.layers), len(net._layer_names))
	print('#blobs =', len(net._blobs), len(net._blob_names))
	print('#blobs =', len(net.blobs))

	#--------------------
	#for layer in net.layers:
	#	print('layer =', type(layer))
	#for blob in net._blobs:
	#	print('blob =', type(blob))

	#print("Layers' names =", net._layer_names)
	#for idx, name in enumerate(net._layer_names):
	#	print('Layer {} = {}'.format(idx, name))
	#print("Blobs' names =", net._blob_names)):
	#for idx, name in enumerate(net._blob_names):
	#	print('Blob {} = {}'.format(idx, name))

	# REF [site] >> http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
	blob_count = 0
	for idx, layer in enumerate(net.layers):
		print('Layer {}: {}, {}, {}'.format(idx, net._layer_names[idx], layer.type, len(layer.blobs)))
		blob_count += len(layer.blobs)
		for ii, blob in enumerate(layer.blobs):
			#print('\tBlob {} = {}, {}'.format(ii, type(blob.data), blob.data.shape))
			blob.data[...] = np.full_like(blob.data, 37)
			print(blob.data)
	print('#actual blobs =', blob_count)

def create_lenet(lmdb, batch_size):
	net = caffe.NetSpec()
	net.data, net.label = caffe.layers.Data(batch_size=batch_size, backend=caffe.params.Data.LMDB, source=lmdb, transform_param=dict(scale=1./255), ntop=2)
	net.conv1 = caffe.layers.Convolution(net.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
	net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
	net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size=2, stride=2, pool=caffe.params.Pooling.MAX)
	net.ip1 = caffe.layers.InnerProduct(net.pool2, num_output=500, weight_filler=dict(type='xavier'))
	net.relu1 = caffe.layers.ReLU(net.ip1, in_place=True)
	net.ip2 = caffe.layers.InnerProduct(net.relu1, num_output=10, weight_filler=dict(type='xavier'))
	net.loss = caffe.layers.SoftmaxWithLoss(net.ip2, net.label)
	return net.to_proto()  # caffe.proto.caffe_pb2.NetParameter.

# REF [site] >> http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
def define_model():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	# ${CAFFE_HOME}/examples/mnist/create_mnist.sh
	train_data_dir_path = caffe_home_dir_path + '/examples/mnist/mnist_train_lmdb'
	test_data_dir_path = caffe_home_dir_path + '/examples/mnist/mnist_test_lmdb'

	# Network definition file.
	train_prototxt_filepath = caffe_home_dir_path + '/examples/mnist/lenet_auto_train.prototxt'
	test_prototxt_filepath = caffe_home_dir_path + '/examples/mnist/lenet_auto_test.prototxt'

	train_model = create_lenet(train_data_dir_path, 64)  # caffe.proto.caffe_pb2.NetParameter.
	with open(train_prototxt_filepath, 'w') as fd:
		fd.write(str(train_model))

	test_model = create_lenet(test_data_dir_path, 100)  # caffe.proto.caffe_pb2.NetParameter.
	with open(test_prototxt_filepath, 'w') as fd:
		fd.write(str(test_model))

# REF [site] >> http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
def create_custom_python_layer():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	model_definition_filepath = './mylayer.prototxt'
	image_filepath = caffe_home_dir_path + '/examples/images/cat_gray.jpg'

	img = np.array(Image.open(image_filepath))
	img_input = img[np.newaxis, np.newaxis, :, :]

	net = caffe.Net(model_definition_filepath, caffe.TEST)

	net.blobs['data'].reshape(*img_input.shape)
	net.blobs['data'].data[...] = img_input
	net.forward()

# REF [site] >> http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
def simple_prediction_example():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	model_definition_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/deploy.prototxt'
	trained_model_weights_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	mean_filepath = caffe_home_dir_path + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'
	label_filepath = caffe_home_dir_path + '/data/ilsvrc12/synset_words.txt'
	input_image_filepath = caffe_home_dir_path + '/examples/images/cat.jpg'

	channel_swap = [2, 1, 0] #None  # RGB -> BGR since BGR is the Caffe default by way of OpenCV.
	raw_scale = 255.0

	# Load the model.
	net = caffe.Net(model_definition_filepath, trained_model_weights_filepath, caffe.TEST)

	# Load input and configure preprocessing.
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
	transformer.set_mean('data', np.load(mean_filepath).mean(1).mean(1))
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_channel_swap('data', channel_swap)
	transformer.set_raw_scale('data', raw_scale)

	# Note we can change the batch size on-the-fly since we classify only one image, we change batch size from 10 to 1.
	net.blobs['data'].reshape(1, 3, 227, 227)

	# Load the image in the data layer.
	img = caffe.io.load_image(input_image_filepath)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)

	# Compute.
	out = net.forward()
	#out = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))

	# Predicted class.
	print('Predicted class =', out['prob'].argmax())

	# Print predicted labels.
	labels = np.loadtxt(label_filepath, str, delimiter='\t')
	top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
	print('Labels =', labels[top_k])

# REF [site] >> http://christopher5106.github.io/deep/learning/2015/09/04/Deep-learning-tutorial-on-Caffe-Technology.html
def simple_train_example():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	solver_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/solver.prototxt'

	solver = caffe.get_solver(solver_filepath)
	#solver = caffe.SGDSolver(solver_filepath)
	#solver = caffe.AdaDelta(solver_filepath)
	#solver = caffe.AdaGrad(solver_filepath)
	#solver = caffe.Adam(solver_filepath)
	#solver = caffe.Nesterov(solver_filepath)
	#solver = caffe.RMSprop(solver_filepath)

	# Now, it's time to begin to see if everything works well and to fill the layers in a forward propagation in the net (computation of net.blobs[k].data from input layer until the loss layer).
	# Trains net.
	solver.net.forward()

	# Tests net (there can be more than one).
	solver.test_nets[0].forward()

	# For the computation of the gradients (computation of the net.blobs[k].diff and net.params[k][j].diff from the loss layer until input layer).
	solver.net.backward()

	# To launch one step of the gradient descent, that is a forward propagation, a backward propagation and the update of the net params given the gradients (update of the net.params[k][j].data).
	solver.step(1)

	# To run the full gradient descent, that is the max_iter steps.
	solver.solve()

	# Computes accuracy of the model on the test data.
	"""
	accuracy = 0
	batch_size = solver.test_nets[0].blobs['data'].num
	test_iters = int(len(Xt) / batch_size)
	for i in range(test_iters):
		solver.test_nets[0].forward()
		accuracy += solver.test_nets[0].blobs['accuracy'].data
	accuracy /= test_iters

	print('Accuracy: {:.3f}'.format(accuracy))
	"""

# REF [file] >> ${CAFFE_HOME}/python/classify.py
def classification_example():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	model_definition_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/deploy.prototxt'
	trained_model_weights_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	mean_filepath = caffe_home_dir_path + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'

	input_image_filepath = caffe_home_dir_path + '/examples/images/cat.jpg'
	output_image_filepath = caffe_home_dir_path + '/examples/images/cat_classification.jpg'

	inputs = [caffe.io.load_image(input_image_filepath)]
	mean = np.load(mean_filepath) if mean_filepath else None

	image_dims = [256, 256]
	center_only = True
	input_scale = 1.0
	raw_scale = 255.0
	channel_swap = [2, 1, 0] #None  # RGB -> BGR since BGR is the Caffe default by way of OpenCV.

	# Make a classifier.
	classifier = caffe.Classifier(
		model_definition_filepath, trained_model_weights_filepath,
		image_dims=image_dims, mean=mean,
		input_scale=input_scale, raw_scale=raw_scale,
		channel_swap=channel_swap
	)

	# Classify.
	start = time.time()
	predictions = classifier.predict(inputs, not center_only)
	print('Done in %.2f s.' % (time.time() - start))

	# REF [site] >> https://github.com/BVLC/caffe/tree/master/examples/cpp_classification
	#	${CAFFE_HOME}/data/ilsvrc12/synset_words.txt
	print('Class ID =', np.argmax(predictions, axis=-1))

	# Save.
	#print('Saving results into %s' % output_image_filepath)
	#np.save(output_image_filepath, predictions)

# REF [file] >> ${CAFFE_HOME}/python/detect.py
def detection_example():
	if 'posix' == os.name:
		caffe_home_dir_path = '/home/sangwook/lib_repo/cpp/caffe_github'
	else:
		caffe_home_dir_path = 'D:/lib_repo/cpp/rnd/caffe_github'
		#caffe_home_dir_path = 'D:/lib_repo/cpp/caffe_github'

	model_definition_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/deploy.prototxt'
	trained_model_weights_filepath = caffe_home_dir_path + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
	mean_filepath = caffe_home_dir_path + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'

	input_image_filepath = caffe_home_dir_path + '/examples/images/cat.jpg'
	output_image_filepath = caffe_home_dir_path + '/examples/images/cat_detection.csv'

	#inputs_for_detect_windows = [(input_image_filepath, np.array([[0, 0, 360, 480]]))]  # If crop_mode = 'list'.
	inputs_for_detect_windows = [(input_image_filepath, np.array([[32, 32, 296, 416]]))]  # If crop_mode = 'list'.
	inputs_for_detect_selective_search = [input_image_filepath]  # If crop_mode = 'selective_search'.

	inputs = [caffe.io.load_image(input_image_filepath)]
	mean = np.load(mean_filepath) if mean_filepath else None

	crop_mode = 'list' #'selective_search'
	input_scale = 1.0
	raw_scale = 255.0
	channel_swap = [2, 1, 0] #None  # RGB -> BGR since BGR is the Caffe default by way of OpenCV.
	context_pad = 16

	# Make a detector.
	detector = caffe.Detector(
		model_definition_filepath, trained_model_weights_filepath, mean=mean,
		input_scale=input_scale, raw_scale=raw_scale,
		channel_swap=channel_swap,
		context_pad=context_pad
	)

	# Detect.
	if 'list' == crop_mode:
		detections = detector.detect_windows(inputs_for_detect_windows)
	else:
		detections = detector.detect_selective_search(inputs_for_detect_selective_search)

	print('**************', type(detections))

	# Save results.
	"""
	t = time.time()
	if args.output_file.lower().endswith('csv'):
		# csv.
		# Enumerate the class probabilities.
		class_cols = ['class{}'.format(x) for x in range(NUM_OUTPUT)]
		df[class_cols] = pd.DataFrame(data=np.vstack(df['feat']), index=df.index, columns=class_cols)
		df.to_csv(args.output_file, cols=COORD_COLS + class_cols)
	else:
		# HDF5.
		df.to_hdf(args.output_file, 'df', mode='w')
	print('Saved to {} in {:.3f} s.'.format(args.output_file, time.time() - t))
	"""

# REF [file] >> ${CAFFE_HOME}/python/train.py
def train_example():
	raise NotImplementedError

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def yolo_object_detection_example():
	if 'posix' == os.name:
		darknet_home_dir_path = '/home/sangwook/lib_repo/cpp/darknet_github'
	else:
		darknet_home_dir_path = 'D:/lib_repo/cpp/rnd/darknet_github'
		#darknet_home_dir_path = 'D:/lib_repo/cpp/darknet_github'

	# Converted YOLOv3 model.
	#	https://github.com/BingzheWu/object_detetction_tools/tree/master/nn_model_transform
	#	https://github.com/BVLC/caffe/pull/6384/commits/4d2400e7ae692b25f034f02ff8e8cd3621725f5c
	deploy_prototxt_filepath = './yolov3.prototxt'  # Network definition file.
	caffe_model_filepath = './yolov3.caffemodel'  # Trained Caffe model file.
	image_filepath = darknet_home_dir_path + '/data/dog.jpg'

	confidence_thresh = 0.5
	hier_thresh = 0.5
	nms_threshold = 0.45

	#--------------------
	img = Image.open(image_filepath)

	# Input image size = (3, 608, 608).
	img = img.resize(size=(608, 608), resample=Image.BICUBIC)
	img = np.asarray(img, dtype=np.uint8)
	# FIXME [check] >> Which one is correct?
	#img = np.transpose(img, (2, 1, 0))
	img = np.transpose(img, (2, 0, 1))
	img = img[::-1,:,:]  # RGB to BGR (for the 1st axis).
	#img = img[:,:,::-1]  # RGB to BGR (for the 3rd axis).
	img = img / 255.0
	#img = np.array(img, order='C')  # C-array type.

	#--------------------
	# CNN reconstruction and loading the trained weights.
	net = caffe.Net(deploy_prototxt_filepath, caffe_model_filepath, caffe.TEST)

	# Each prediction composes of a boundary box, a objectness, and 80 class scores, N * N * (B * (4 + 1 + C)).
	#	N: the number of grids (19x19, 38x38, 76x76). 608 / 19 = 32, 608 / 38 = 16, 608 / 76 = 8.
	#	B: the number of bounding boxes a cell on the feature map can predict. In case of COCO, B = 3.
	#	4: 4 bounding box attributes.
	#	1: one object confidence.
	#	C: the number of classes. In case of COCO, C = 80.
	yolo_outputs = net.forward_all(data=np.asarray([img]))

	print("YOLO outputs' shapes =", yolo_outputs['layer83-yolo'].shape, yolo_outputs['layer95-yolo'].shape, yolo_outputs['layer107-yolo'].shape)

	#--------------------
	# REF [function] >> yolo_object_detection() in ${SWDT_PYTHON_HOME}/rnd/test/machine_vision/opencv/opencv_dnn.py
	bbox_count = 0
	for outp in yolo_outputs.values():
		outp = np.reshape(outp, outp.shape[:2] + (-1,))
		outp = np.dstack((outp[:,0:85,:], outp[:,85:170,:], outp[:,170:,:]))
		#outp = np.concatenate((outp[:,0:85,:], outp[:,85:170,:], outp[:,170:,:]), axis=-1)
		# FIXME [fix] >> Implementation is not finished.
		outp = sigmoid(outp)
		#outp = np.exp(sigmoid(outp))

		for dd in range(outp.shape[2]):
			scores = outp[0,5:,dd]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > confidence_thresh:
				++bbox_count
	print('+++++++++++++++', bbox_count)

	aa = yolo_outputs['layer83-yolo']
	print(aa[0,:4,0,0])  # Bounding box. (?)
	print(aa[0,4,0,0])  # Objectness. (?)
	print(aa[0,5:85,0,0])  # Class scores. (?)
	aa = sigmoid(yolo_outputs['layer83-yolo'])
	print(aa[0,:4,0,0])  # Bounding box. (?)
	print(aa[0,4,0,0])  # Objectness. (?)
	print(aa[0,5:85,0,0])  # Class scores. (?)
	aa = np.exp(sigmoid(yolo_outputs['layer83-yolo']))
	print(aa[0,:4,0,0])  # Bounding box. (?)
	print(aa[0,4,0,0])  # Objectness. (?)
	print(aa[0,5:85,0,0])  # Class scores. (?)

def main():
	#caffe.set_mode_cpu()
	caffe.set_device(0)
	caffe.set_mode_gpu()

	#parse_caffe_model()
	#parse_caffe_prototxt()

	#numpy_to_blobproto()
	load_npy_weights_into_caffe_model()

	#define_model()
	#create_custom_python_layer()

	#simple_prediction_example()
	#simple_train_example()

	#classification_example()
	#detection_example()  # Not working.
	#train_example()  # Not yet implemented.

	#yolo_object_detection_example()

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
