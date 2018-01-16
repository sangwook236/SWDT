# REF [site] >> https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import tensorflow as tf

#%%------------------------------------------------------------------
# Create a VGG model.

def vgg16(inputs):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
			activation_fn=tf.nn.relu,
			weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
			weights_regularizer=slim.l2_regularizer(0.0005)):
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		net = slim.max_pool2d(net, [2, 2], scope='pool3')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		net = slim.max_pool2d(net, [2, 2], scope='pool5')
		net = slim.fully_connected(net, 4096, scope='fc6')
		net = slim.dropout(net, 0.5, scope='dropout6')
		net = slim.fully_connected(net, 4096, scope='fc7')
		net = slim.dropout(net, 0.5, scope='dropout7')
		net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
		return net

#%%------------------------------------------------------------------

vgg = nets.vgg

# Load the images and labels.
images, labels = ...

# Create the model.
predictions, _ = vgg.vgg_16(images)

# Define the loss functions and get the total loss.
loss = slim.losses.softmax_cross_entropy(predictions, labels)

#%%------------------------------------------------------------------

# Load the images and labels.
images, scene_labels, depth_labels = ...

# Create the model.
scene_predictions, depth_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)

# The following two lines have the same effect:
total_loss = classification_loss + sum_of_squares_loss
total_loss = slim.losses.get_total_loss(add_regularization_losses=False)

#%%------------------------------------------------------------------

# Load the images and labels.
images, scene_labels, depth_labels, pose_labels = ...

# Create the model.
scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)

# Define the loss functions and get the total loss.
classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
slim.losses.add_loss(pose_loss)  # Letting TF-Slim know about the additional loss.

# The following two ways to compute the total loss are equivalent:
regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss

# (Regularization Loss is included in the total loss by default).
total_loss2 = slim.losses.get_total_loss()

#%%------------------------------------------------------------------

train_log_dir = ...
if not tf.gfile.Exists(train_log_dir):
	tf.gfile.MakeDirs(train_log_dir)

with tf.Graph().as_default():
	# Set up the data loading:
	images, labels = ...

	# Define the model:
	predictions = vgg.vgg_16(images, is_training=True)

	# Specify the loss function:
	slim.losses.softmax_cross_entropy(predictions, labels)

	total_loss = slim.losses.get_total_loss()
	tf.summary.scalar('losses/total_loss', total_loss)

	# Specify the optimization scheme:
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)

	# create_train_op that ensures that when we evaluate it to get the loss,
	# the update_ops are done and the gradient updates are computed.
	train_tensor = slim.learning.create_train_op(total_loss, optimizer)

	# Actually runs training.
	slim.learning.train(train_tensor, train_log_dir, number_of_steps=1000, save_summaries_secs=300, save_interval_secs=600)
