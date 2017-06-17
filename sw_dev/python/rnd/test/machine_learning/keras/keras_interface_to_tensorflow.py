# REF [site] >> https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import InputLayer
from keras.models import Sequential
from keras.models import model_from_config
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy

#%%-------------------------------------------------------------------

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# This means that Keras will use the session we registered to initialize all variables that it creates internally.
K.set_session(sess)
K.set_learning_phase(0)

#%%-------------------------------------------------------------------

mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

#%%-------------------------------------------------------------------
# Call Keras layers on TensorFlow tensors.

# This placeholder will contain our input digits, as flat vectors.
img = tf.placeholder(tf.float32, shape = (None, 784))

# Use Keras layers to speed up the model definition process.
# Keras layers can be called on TensorFlow tensors.
x = Dense(128, activation = 'relu')(img)  # Fully-connected layer with 128 units and ReLU activation.
x = Dense(128, activation = 'relu')(x)
preds = Dense(10, activation = 'softmax')(x)  # Output layer with 10 units and a softmax activation.

# Define the placeholder for the labels, and the loss function.
labels = tf.placeholder(tf.float32, shape = (None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# Train the model with a TensorFlow optimizer.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# Initialize all variables.
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop.
with sess.as_default():
	for i in range(100):
		batch = mnist_data.train.next_batch(50)
		train_step.run(feed_dict = {img: batch[0], labels: batch[1]})

# Evaluate the model.
acc_value = accuracy(labels, preds)
with sess.as_default():
	print acc_value.eval(feed_dict = {img: mnist_data.test.images, labels: mnist_data.test.labels})

#%%-------------------------------------------------------------------
# Different behaviors during training and testing.

# The Keras learning phase (a scalar TensorFlow tensor) is accessible via the Keras backend.
print(K.learning_phase())

# To make use of the learning phase, simply pass the value "1" (training mode) or "0" (test mode) to feed_dict.
# Train mode.
train_step.run(feed_dict = {x: batch[0], labels: batch[1], K.learning_phase(): 1})

img = tf.placeholder(tf.float32, shape = (None, 784))
labels = tf.placeholder(tf.float32, shape = (None, 10))

x = Dense(128, activation = 'relu')(img)
x = Dropout(0.5)(x)
x = Dense(128, activation = 'relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation = 'softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with sess.as_default():
	for i in range(100):
		batch = mnist_data.train.next_batch(50)
		train_step.run(feed_dict = {img: batch[0], labels: batch[1], K.learning_phase(): 1})

acc_value = accuracy(labels, preds)
with sess.as_default():
	print(acc_value.eval(feed_dict = {img: mnist_data.test.images, labels: mnist_data.test.labels, K.learning_phase(): 0}))

#%%-------------------------------------------------------------------
% Compatibility with name scopes, device scopes.

# Keras layers and models are fully compatible with TensorFlow name scopes.
x = tf.placeholder(tf.float32, shape = (None, 20, 64))
with tf.name_scope('block1'):
	y = LSTM(32, name = 'mylstm')(x)

# Device scope.
with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32, shape = (None, 20, 64))
	y = LSTM(32)(x)  # All ops/variables in the LSTM layer will live on GPU:0.

# Compatibility with graph scopes.
# Any Keras layer or model that you define inside a TensorFlow graph scope will have all of its variables and operations created as part of the specified graph.
my_graph = tf.Graph()
with my_graph.as_default():
	x = tf.placeholder(tf.float32, shape = (None, 20, 64))
	y = LSTM(32)(x)  # All ops/variables in the LSTM layer are created as part of our graph.

# Compatibility with variable scopes.
lstm = LSTM(32)

# Instantiate two TF placeholders.
x = tf.placeholder(tf.float32, shape = (None, 20, 64))
y = tf.placeholder(tf.float32, shape = (None, 20, 64))

# Encode the two tensors with the *same* LSTM weights.
x_encoded = lstm(x)
y_encoded = lstm(y)

# Collect trainable weights and state updates.
# Some Keras layers (stateful RNNs and BatchNormalization layers) have internal updates that need to be run as part of each training step.
layer = BatchNormalization()(x)

update_ops = []
for old_value, new_value in layer.updates:
	update_ops.append(tf.assign(old_value, new_value))

# In case you need to explicitly collect a layer's trainable weights, you can do so via layer.trainable_weights (or model.trainable_weights), a list of TensorFlow Variable instances.
layer = Dense(32)(x)  # Instantiate and call a layer.
print layer.trainable_weights  # List of TensorFlow Variables.

#%%-------------------------------------------------------------------
# Use Keras models with TensorFlow.

# Convert a Keras Sequential model for use in a TensorFlow workflow.
model = Sequential()
model.add(Dense(32, activation = 'relu', input_dim = 784))
model.add(Dense(10, activation = 'softmax'))

# This is our modified Keras model.
model = Sequential()
model.add(InputLayer(input_tensor = custom_input_tensor, input_shape = (None, 784)))
# Build the rest of the model as before.
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
output_tensor = model.output

# Call a Keras model on a TensorFlow tensor.
model = Sequential()
model.add(Dense(32, activation = 'relu', input_dim = 784))
model.add(Dense(10, activation = 'softmax'))

x = tf.placeholder(tf.float32, shape = (None, 784))
y = model(x)

#%%-------------------------------------------------------------------
# Multi-GPU and distributed training.

# Assign part of a Keras model to different GPUs.
with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32, shape = (None, 20, 64))
	y = LSTM(32)(x)  # All ops in the LSTM layer will live on GPU:0.

with tf.device('/gpu:1'):
	x = tf.placeholder(tf.float32, shape = (None, 20, 64))
	y = LSTM(32)(x)  # All ops in the LSTM layer will live on GPU:1.

# If you want to train multiple replicas of a same model on different GPUs, while sharing the same weights across the different replicas, you should first instantiate your model (or layers) under one device scope, then call the same model instance multiple times in different GPU device scopes.
with tf.device('/cpu:0'):
	x = tf.placeholder(tf.float32, shape = None, 784))

	# Shared model living on CPU:0.
	# It won't actually be run during training; it acts as an op template and as a repository for shared variables.
	model = Sequential()
	model.add(Dense(32, activation = 'relu', input_dim = 784))
	model.add(Dense(10, activation = 'softmax'))

# Replica 0
with tf.device('/gpu:0'):
	output_0 = model(x)  # All ops in the replica will live on GPU:0.

# Replica 1.
with tf.device('/gpu:1'):
	output_1 = model(x)  # All ops in the replica will live on GPU:1.

# Merge outputs on CPU.
with tf.device('/cpu:0'):
	preds = 0.5 * (output_0 + output_1)

# We only run the 'preds' tensor, so that only the two replicas on GPU get run (plus the merge op on CPU).
output_value = sess.run([preds], feed_dict = {x: data})

# Distributed training.
# You can trivially make use of TensorFlow distributed training by registering with Keras a TF session linked to a cluster.
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)

K.set_session(sess)

#%%-------------------------------------------------------------------
# Export a model with TensorFlow-serving.

# Two simple steps in action.
K.set_learning_phase(0)  # All new operations will be in test mode from now on.

# Serialize the model and get its weights, for quick re-building.
config = previous_model.get_config()
weights = previous_model.get_weights()

# Re-build a model where the learning phase is now hard-coded to 0.
new_model = model_from_config(config)
new_model.set_weights(weights)

# We can now use TensorFlow-serving to export the model.
export_path = '???'  # Where to save the exported graph.
export_version = 000000  # Version number (integer).

saver = tf.train.Saver(sharded = True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor = model.input, scores_tensor = model.output)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature = signature)
model_exporter.export(export_path, tf.constant(export_version), sess)
