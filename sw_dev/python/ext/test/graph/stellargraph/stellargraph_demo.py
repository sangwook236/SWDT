#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/index.html

#import os
import pandas as pd
import stellargraph as sg
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
#from IPython.display import display, HTML
#%matplotlib inline

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/gcn-link-prediction.html
def predict_link_with_GCN():
	# Load the CORA network data.
	dataset = sg.datasets.Cora()
	#display(HTML(dataset.description))
	print(dataset.description)

	G, _ = dataset.load(subject_as_feature=True)
	print(G.info())

	# Define an edge splitter on the original graph G:
	edge_splitter_test = sg.data.EdgeSplitter(G)

	# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
	# reduced graph G_test with the sampled links removed:
	G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
		p=0.1, method='global', keep_connected=True
	)

	# Define an edge splitter on the reduced graph G_test:
	edge_splitter_train = sg.data.EdgeSplitter(G_test)

	# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
	# reduced graph G_train with the sampled links removed:
	G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
		p=0.1, method='global', keep_connected=True
	)

	#--------------------
	# Create the GCN link model.
	train_gen = sg.mapper.FullBatchLinkGenerator(G_train, method='gcn')
	test_gen = sg.mapper.FullBatchLinkGenerator(G_test, method='gcn')

	gcn = sg.layer.GCN(layer_sizes=[16, 16], activations=['relu', 'relu'], generator=train_gen, dropout=0.3)
	x_inp, x_out = gcn.in_out_tensors()

	prediction = sg.layer.LinkEmbedding(activation='relu', method='ip')(x_out)
	prediction = tf.keras.layers.Reshape((-1,))(prediction)

	model = tf.keras.Model(inputs=x_inp, outputs=prediction)

	#--------------------
	# Train the model.
	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.01),
		loss=tf.keras.losses.binary_crossentropy,
		metrics=['acc'],
	)

	train_flow = train_gen.flow(edge_ids_train, edge_labels_train)
	test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

	if True:
		# Evaluate the initial (untrained) model.
		init_train_metrics = model.evaluate(train_flow)
		init_test_metrics = model.evaluate(test_flow)

		print('\nTrain Set Metrics of the initial (untrained) model:')
		for name, val in zip(model.metrics_names, init_train_metrics):
			print('\t{}: {:0.4f}'.format(name, val))

		print('\nTest Set Metrics of the initial (untrained) model:')
		for name, val in zip(model.metrics_names, init_test_metrics):
			print('\t{}: {:0.4f}'.format(name, val))

	epochs = 50
	history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False)
	sg.utils.plot_history(history)

	#--------------------
	# Evaluate the trained model.
	train_metrics = model.evaluate(train_flow)
	test_metrics = model.evaluate(test_flow)

	print('\nTrain Set Metrics of the trained model:')
	for name, val in zip(model.metrics_names, train_metrics):
		print('\t{}: {:0.4f}'.format(name, val))

	print('\nTest Set Metrics of the trained model:')
	for name, val in zip(model.metrics_names, test_metrics):
		print('\t{}: {:0.4f}'.format(name, val))

	plt.show()

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html
def classify_node_with_GCN():
	# Load the CORA network.
	dataset = sg.datasets.Cora()
	#display(HTML(dataset.description))
	print(dataset.description)

	G, node_subjects = dataset.load()
	print(G.info())
	print(node_subjects.value_counts().to_frame())

	# Split the data.
	train_subjects, test_subjects = sklearn.model_selection.train_test_split(
		node_subjects, train_size=140, test_size=None, stratify=node_subjects
	)
	val_subjects, test_subjects = sklearn.model_selection.train_test_split(
		test_subjects, train_size=500, test_size=None, stratify=test_subjects
	)
	print(train_subjects.value_counts().to_frame())

	# Convert to numeric arrays.
	target_encoding = sklearn.preprocessing.LabelBinarizer()

	train_targets = target_encoding.fit_transform(train_subjects)
	val_targets = target_encoding.transform(val_subjects)
	test_targets = target_encoding.transform(test_subjects)

	#--------------------
	# Create the GCN layers.
	generator = sg.mapper.FullBatchNodeGenerator(G, method='gcn')
	gcn = sg.layer.GCN(layer_sizes=[16, 16], activations=['relu', 'relu'], generator=generator, dropout=0.5)
	x_inp, x_out = gcn.in_out_tensors()

	predictions = tf.keras.layers.Dense(units=train_targets.shape[1], activation='softmax')(x_out)

	model = tf.keras.Model(inputs=x_inp, outputs=predictions)

	#--------------------
	# Train the model.
	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.01),
		loss=tf.keras.losses.categorical_crossentropy,
		metrics=['acc'],
	)

	train_gen = generator.flow(train_subjects.index, train_targets)
	val_gen = generator.flow(val_subjects.index, val_targets)
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True)

	history = model.fit(
		train_gen,
		epochs=200,
		validation_data=val_gen,
		verbose=2,
		shuffle=False,  # This should be False, since shuffling data means shuffling the whole graph.
		callbacks=[es_callback],
	)
	sg.utils.plot_history(history)

	#--------------------
	# Evaluate the model.
	test_gen = generator.flow(test_subjects.index, test_targets)
	test_metrics = model.evaluate(test_gen)
	print('\nTest Set Metrics:')
	for name, val in zip(model.metrics_names, test_metrics):
		print('\t{}: {:0.4f}'.format(name, val))

	#--------------------
	# Make predictions with the model.
	all_nodes = node_subjects.index
	all_gen = generator.flow(all_nodes)
	all_predictions = model.predict(all_gen)

	node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

	df = pd.DataFrame({'Predicted': node_predictions, 'True': node_subjects})
	print(df.head(20))

	#--------------------
	# Node embedding.
	embedding_model = tf.keras.Model(inputs=x_inp, outputs=x_out)
	emb = embedding_model.predict(all_gen)
	print(emb.shape)

	from sklearn.decomposition import PCA
	from sklearn.manifold import TSNE

	transform = TSNE  # or PCA.

	X = emb.squeeze(0)
	print(X.shape)

	trans = transform(n_components=2)
	X_reduced = trans.fit_transform(X)
	print(X_reduced.shape)

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.scatter(
		X_reduced[:, 0],
		X_reduced[:, 1],
		c=node_subjects.astype('category').cat.codes,
		cmap='jet',
		alpha=0.7,
	)
	ax.set(
		aspect='equal',
		xlabel='$X_1$',
		ylabel='$X_2$',
		title=f'{transform.__name__} visualization of GCN embeddings for cora dataset',
	)
	plt.show()

def main():
	# Link prediction.
	predict_link_with_GCN()

	# Node classification.
	#classify_node_with_GCN()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
