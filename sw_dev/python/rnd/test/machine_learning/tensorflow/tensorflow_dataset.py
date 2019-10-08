#!/usr/bin/env python

import numpy as np
import tensorflow as tf

# REF [site] >> https://www.tensorflow.org/guide/datasets
def dataset_basic():
	dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
	print(dataset1.output_types)  # ==> 'tf.float32'
	print(dataset1.output_shapes)  # ==> '(10,)'

	dataset2 = tf.data.Dataset.from_tensor_slices(
		(tf.random_uniform([4]),
		 tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
	print(dataset2.output_types)  # ==> '(tf.float32, tf.int32)'
	print(dataset2.output_shapes)  # ==> '((), (100,))'

	dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
	print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
	print(dataset3.output_shapes)  # ==> '(10, ((), (100,)))'

	dataset4 = tf.data.Dataset.from_tensor_slices(
		{'a': tf.random_uniform([4]),
		 'b': tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
	print(dataset4.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
	print(dataset4.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

	#--------------------
	"""
	dataset1 = dataset1.map(lambda x: ...)
	dataset2 = dataset2.flat_map(lambda x, y: ...)
	# Note: Argument destructuring is not available in Python 3.
	dataset3 = dataset3.filter(lambda x, (y, z): ...)
	"""

	#--------------------
	# One-shot iterator.

	sess = tf.Session()

	dataset = tf.data.Dataset.range(100)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()

	for i in range(100):
		value = sess.run(next_element)
		print('{} == {}'.format(value, i))

	#--------------------
	# Initializable iterator.

	max_value = tf.placeholder(tf.int64, shape=[])
	dataset = tf.data.Dataset.range(max_value)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	print('Start training with an initializable iterator...')
	for _ in range(20):
		# Initialize an iterator over a dataset with 10 elements.
		sess.run(iterator.initializer, feed_dict={max_value: 10})
		for i in range(10):
			value = sess.run(next_element)
			#print('{} == {}'.format(value, i))

		# Initialize the same iterator over a dataset with 100 elements.
		sess.run(iterator.initializer, feed_dict={max_value: 100})
		for i in range(100):
			value = sess.run(next_element)
			#print('{} == {}'.format(value, i))
	print('End training with an initializable iterator.')

	#--------------------
	# Reinitializable iterator.

	max_value = tf.placeholder(tf.int64, shape=[])
	training_dataset = tf.data.Dataset.range(max_value).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
	validation_dataset = tf.data.Dataset.range(max_value)

	iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
	next_element = iterator.get_next()

	training_init_op = iterator.make_initializer(training_dataset)
	validation_init_op = iterator.make_initializer(validation_dataset)

	print('Start training with a reinitializable iterator...')
	# Run 20 epochs in which the training dataset is traversed, followed by the validation dataset.
	for _ in range(20):
		# Initialize an iterator over the training dataset.
		sess.run(training_init_op, feed_dict={max_value: 100})
		for _ in range(100):
			sess.run(next_element)

		# Initialize an iterator over the validation dataset.
		sess.run(validation_init_op, feed_dict={max_value: 100})
		for _ in range(50):
			sess.run(next_element)
	print('End training with a reinitializable iterator.')

	#--------------------
	# Feedable iterator.

	training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
	validation_dataset = tf.data.Dataset.range(50)

	handle = tf.placeholder(tf.string, shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
	next_element = iterator.get_next()

	training_iterator = training_dataset.make_one_shot_iterator()
	validation_iterator = validation_dataset.make_initializable_iterator()

	# The 'Iterator.string_handle()' method returns a tensor that can be evaluated and used to feed the 'handle' placeholder.
	training_handle = sess.run(training_iterator.string_handle())
	validation_handle = sess.run(validation_iterator.string_handle())

	print('Start training with a feedable iterator...')
	# Loop forever, alternating between training and validation.
	#while True:
	for _ in range(20):
		# Run 200 steps using the training dataset.
		# Note that the training dataset is nfinite, and we resume from where we left off in the previous 'while' loop iteration.
		for _ in range(200):
			sess.run(next_element, feed_dict={handle: training_handle})

		# Run one pass over the validation dataset.
		sess.run(validation_iterator.initializer)
		for _ in range(50):
			sess.run(next_element, feed_dict={handle: validation_handle})
	print('End training with a feedable iterator.')

	#--------------------
	# Consuming values from an iterator.

	dataset = tf.data.Dataset.range(5)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	# Typically 'result' will be the output of a model, or an optimizer's training operation.
	result = tf.add(next_element, next_element)

	sess.run(iterator.initializer)
	if False:
		print(sess.run(result))  # ==> "0"
		print(sess.run(result))  # ==> "2"
		print(sess.run(result))  # ==> "4"
		print(sess.run(result))  # ==> "6"
		print(sess.run(result))  # ==> "8"
		try:
			sess.run(result)
		except tf.errors.OutOfRangeError:
			print('End of dataset')  # ==> "End of dataset"
	else:
		while True:
			try:
				print(sess.run(result))
			except tf.errors.OutOfRangeError:
				break

	"""
	# If each element of the dataset has a nested structure, the return value of Iterator.get_next() will be one or more tf.Tensor objects in the same nested structure:
	dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
	dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
	dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

	iterator = dataset3.make_initializable_iterator()

	sess.run(iterator.initializer)
	next1, (next2, next3) = iterator.get_next()
	"""

# REF [site] >> https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
def dataset_example():
	#--------------------
	# Import data.

	# From numpy.
	x = np.random.sample((100, 2))
	dataset = tf.data.Dataset.from_tensor_slices(x)

	features, labels = (np.random.sample((100, 2)), np.random.sample((100, 1)))
	dataset = tf.data.Dataset.from_tensor_slices((features, labels))

	# From tensors.
	dataset = tf.data.Dataset.from_tensor_slices(tf.random_uniform([100, 2]))

	# From a placeholder.
	x = tf.placeholder(tf.float32, shape=[None, 2])
	dataset = tf.data.Dataset.from_tensor_slices(x)

	# From generator.
	if False:
		def gen():
			for i in range(10):
				yield (i, [1] * i)

		dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
	else:
		sequence = np.array([[[1]], [[2], [3]], [[3], [4], [5]]])
		def generator():
			for el in sequence:
				yield el

		#dataset = tf.data.Dataset().batch(1).from_generator(generator, output_types=tf.int64, output_shapes=(tf.TensorShape([None, 1])))
		dataset = tf.data.Dataset.from_generator(generator, output_types=tf.int64, output_shapes=(tf.TensorShape([None, 1])))

	iter = dataset.make_initializable_iterator()
	el = iter.get_next()

	with tf.Session() as sess:
		sess.run(iter.initializer)
		print(sess.run(el))
		print(sess.run(el))
		print(sess.run(el))

	# From CSV file.
	CSV_PATH = './tweets.csv'
	dataset = tf.data.experimental.make_csv_dataset(CSV_PATH, batch_size=32)
	iter = dataset.make_one_shot_iterator()
	next = iter.get_next()
	print(next)  # next is a dict with key=columns names and value=column data.
	inputs, labels = next['text'], next['sentiment']

	with tf.Session() as sess:
		sess.run([inputs, labels])

	#--------------------
	# Create an iterator.

	# One shot iterator.
	x = np.random.sample((100, 2))
	dataset = tf.data.Dataset.from_tensor_slices(x)
	iter = dataset.make_one_shot_iterator()
	el = iter.get_next()

	with tf.Session() as sess:
		print(sess.run(el))

	# Initializable iterator.
	x = tf.placeholder(tf.float32, shape=[None, 2])
	dataset = tf.data.Dataset.from_tensor_slices(x)
	iter = dataset.make_initializable_iterator()
	el = iter.get_next()

	data = np.random.sample((100, 2))

	with tf.Session() as sess:
		sess.run(iter.initializer, feed_dict={x: data}) 
		print(sess.run(el))

	x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
	dataset = tf.data.Dataset.from_tensor_slices((x, y))
	iter = dataset.make_initializable_iterator()
	features, labels = iter.get_next()

	train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
	test_data = (np.array([[1, 2]]), np.array([[0]]))

	EPOCHS = 10
	with tf.Session() as sess:
		# Initialize iterator with train data.
		sess.run(iter.initializer, feed_dict={x: train_data[0], y: train_data[1]})
		for _ in range(EPOCHS):
			sess.run([features, labels])
		# Switch to test data.
		sess.run(iter.initializer, feed_dict={x: test_data[0], y: test_data[1]})
		print(sess.run([features, labels]))

	# Reinitializable iterator.
	train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
	test_data = (np.random.sample((10, 2)), np.random.sample((10, 1)))

	train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
	test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
	iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
	features, labels = iter.get_next()
	train_init_op = iter.make_initializer(train_dataset)
	test_init_op = iter.make_initializer(test_dataset)

	EPOCHS = 10
	with tf.Session() as sess:
		sess.run(train_init_op)  # Switch to train dataset.
		for _ in range(EPOCHS):
			sess.run([features, labels])
		sess.run(test_init_op)  # Switch to val dataset.
		print(sess.run([features, labels]))

	# Feedable iterator.
	# This is very similar to the reinitializable iterator, but instead of switching between datasets, it switches between iterators.
	train_data = (np.random.sample((100, 2)), np.random.sample((100, 1)))
	test_data = (np.random.sample((10, 2)), np.random.sample((10, 1)))

	x, y = tf.placeholder(tf.float32, shape=[None, 2]), tf.placeholder(tf.float32, shape=[None, 1])
	train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
	test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
	train_iterator = train_dataset.make_initializable_iterator()
	test_iterator = test_dataset.make_initializable_iterator()
	# Same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
	handle = tf.placeholder(tf.string, shape=[])
	iter = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
	next_elements = iter.get_next()

	EPOCHS = 10
	with tf.Session() as sess:
		train_handle = sess.run(train_iterator.string_handle())
		test_handle = sess.run(test_iterator.string_handle())

		sess.run(train_iterator.initializer, feed_dict={x: train_data[0], y: train_data[1]})
		sess.run(test_iterator.initializer, feed_dict={x: test_data[0], y: test_data[1]})

		for _ in range(EPOCHS):
			x,y = sess.run(next_elements, feed_dict = {handle: train_handle})
			print(x, y)

		x,y = sess.run(next_elements, feed_dict = {handle: test_handle})
		print(x, y)

	#--------------------
	# Consume data.

	EPOCHS, BATCH_SIZE = 10, 16

	features, labels = (np.array([np.random.sample((100, 2))]), np.array([np.random.sample((100, 1))]))

	dataset = tf.data.Dataset.from_tensor_slices((features, labels)).repeat().batch(BATCH_SIZE)
	iter = dataset.make_one_shot_iterator()
	x, y = iter.get_next()

	# Make a simple model.
	net = tf.layers.dense(x, 8, activation=tf.tanh)  # Pass the first value from iter.get_next() as input,
	net = tf.layers.dense(net, 8, activation=tf.tanh)
	prediction = tf.layers.dense(net, 1, activation=tf.tanh)

	loss = tf.losses.mean_squared_error(prediction, y)  # Pass the second value from iter.get_next() as label,
	train_op = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(EPOCHS):
			_, loss_value = sess.run([train_op, loss])
			print('Iter: {}, Loss: {:.4f}'.format(i, loss_value))

	#--------------------
	# batch(), repeat(), shuffle().

	BATCH_SIZE = 4
	x = np.random.sample((100, 2))

	dataset = tf.data.Dataset.from_tensor_slices(x)
	dataset = dataset.shuffle(buffer_size=100)
	#dataset = dataset.repeat()
	dataset = dataset.batch(BATCH_SIZE)
	iter = dataset.make_one_shot_iterator()
	el = iter.get_next()

	with tf.Session() as sess:
		print(sess.run(el))

	#--------------------
	# map(), flat_map(), filter().

	if True:
		x = np.array([[1], [2], [3], [4]])

		dataset = tf.data.Dataset.from_tensor_slices(x)
		dataset = dataset.map(lambda x: x * 2)
	else:
		x, y = np.array([[1], [2], [3], [4]]), np.array([[1], [2], [3], [4]])

		dataset = tf.data.Dataset.from_tensor_slices((x, y))
		dataset = dataset.map(lambda x, y: (x * 2, y**2))
	iter = dataset.make_one_shot_iterator()
	el = iter.get_next()

	with tf.Session() as sess:
		for _ in range(len(x)):
			print(sess.run(el))

def main():
	dataset_basic()
	#dataset_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
