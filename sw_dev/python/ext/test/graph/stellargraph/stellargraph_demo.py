#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/index.html

import os
import numpy as np
import pandas as pd
import stellargraph as sg
import tensorflow as tf
import sklearn, sklearn.manifold
import matplotlib.pyplot as plt
#from IPython.display import display, HTML
#%matplotlib inline

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/graph-classification/gcn-supervised-graph-classification.html
def supervised_graph_classification_with_GCN():
	# Import the data.
	dataset = sg.datasets.MUTAG()
	#display(HTML(dataset.description))
	print(dataset.description)

	graphs, graph_labels = dataset.load()
	print(graphs[0].info())
	print(graphs[1].info())

	summary = pd.DataFrame(
		[(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
		columns=['nodes', 'edges'],
	)
	print(summary.describe().round(1))
	print(graph_labels.value_counts().to_frame())

	graph_labels = pd.get_dummies(graph_labels, drop_first=True)

	# Prepare graph generator.
	generator = sg.mapper.PaddedGraphGenerator(graphs=graphs)

	#--------------------
	# Create the Keras graph classification model.
	def create_graph_classification_model(generator):
		gc_model = sg.layer.GCNSupervisedGraphClassification(
			layer_sizes=[64, 64],
			activations=['relu', 'relu'],
			generator=generator,
			dropout=0.5,
		)
		x_inp, x_out = gc_model.in_out_tensors()
		predictions = tf.keras.layers.Dense(units=32, activation='relu')(x_out)
		predictions = tf.keras.layers.Dense(units=16, activation='relu')(predictions)
		predictions = tf.keras.layers.Dense(units=1, activation='sigmoid')(predictions)

		# Create the Keras model and prepare it for training.
		model = tf.keras.Model(inputs=x_inp, outputs=predictions)
		model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])

		return model

	#--------------------
	# Train the model.
	epochs = 200  # Maximum number of training epochs.
	folds = 10  # The number of folds for k-fold cross validation.
	n_repeats = 5  # The number of repeats for repeated k-fold cross validation.

	es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=25, restore_best_weights=True)

	def train_fold(model, train_gen, test_gen, es, epochs):
		history = model.fit(train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es])

		# Calculate performance on the test data and return along with history.
		test_metrics = model.evaluate(test_gen, verbose=0)
		test_acc = test_metrics[model.metrics_names.index('acc')]

		return history, test_acc

	def get_generators(train_index, test_index, graph_labels, batch_size):
		train_gen = generator.flow(train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size)
		test_gen = generator.flow(test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size)

		return train_gen, test_gen

	stratified_folds = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=folds, n_repeats=n_repeats).split(graph_labels, graph_labels)

	test_accs = list()
	for i, (train_index, test_index) in enumerate(stratified_folds):
		print(f'Training and evaluating on fold {i+1} out of {folds * n_repeats}...')
		train_gen, test_gen = get_generators(train_index, test_index, graph_labels, batch_size=30)

		model = create_graph_classification_model(generator)

		history, acc = train_fold(model, train_gen, test_gen, es, epochs)

		test_accs.append(acc)

	print(f'Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%')

	plt.figure(figsize=(8, 6))
	plt.hist(test_accs)
	plt.xlabel('Accuracy')
	plt.ylabel('Count')
	plt.show()

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/link-prediction/gcn-link-prediction.html
def link_prediction_with_GCN():
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
def node_classification_with_GCN():
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

	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.01),
		loss=tf.keras.losses.categorical_crossentropy,
		metrics=['acc'],
	)

	#--------------------
	# Train the model.
	train_gen = generator.flow(train_subjects.index, train_targets)
	val_gen = generator.flow(val_subjects.index, val_targets)
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True)

	epochs = 200
	history = model.fit(
		train_gen,
		epochs=epochs,
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

	#transform = sklearn.decomposition.PCA
	transform = sklearn.manifold.TSNE

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

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gat-node-classification.html
def node_classification_with_GAT():
	# Load the CORA network.
	dataset = sg.datasets.Cora()
	#display(HTML(dataset.description))
	print(dataset.description)

	G, node_subjects = dataset.load()
	print(G.info())
	print(set(node_subjects))

	# Split the data.
	train_subjects, test_subjects = sklearn.model_selection.train_test_split(
		node_subjects, train_size=140, test_size=None, stratify=node_subjects
	)
	val_subjects, test_subjects = sklearn.model_selection.train_test_split(
		test_subjects, train_size=500, test_size=None, stratify=test_subjects
	)
	from collections import Counter
	print(Counter(train_subjects))

	# Convert to numeric arrays.
	target_encoding = sklearn.preprocessing.LabelBinarizer()

	train_targets = target_encoding.fit_transform(train_subjects)
	val_targets = target_encoding.transform(val_subjects)
	test_targets = target_encoding.transform(test_subjects)

	#--------------------
	# Create the GAT model.
	generator = sg.mapper.FullBatchNodeGenerator(G, method='gat')
	gat = sg.layer.GAT(
		layer_sizes=[8, train_targets.shape[1]],
		activations=['elu', 'softmax'],
		attn_heads=8,
		generator=generator,
		in_dropout=0.5,
		attn_dropout=0.5,
		normalize=None,
	)
	x_inp, predictions = gat.in_out_tensors()

	model = tf.keras.Model(inputs=x_inp, outputs=predictions)

	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.005),
		loss=tf.keras.losses.categorical_crossentropy,
		metrics=['acc'],
	)

	#--------------------
	# Train the model.
	train_gen = generator.flow(train_subjects.index, train_targets)
	val_gen = generator.flow(val_subjects.index, val_targets)

	if not os.path.isdir('logs'):
		os.makedirs('logs')
	es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=20)  # patience is the number of epochs to wait before early stopping in case of no further improvement.
	mc_callback = tf.keras.callbacks.ModelCheckpoint('logs/best_model.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)

	epochs = 50
	history = model.fit(
		train_gen,
		epochs=epochs,
		validation_data=val_gen,
		verbose=2,
		shuffle=False,  # This should be False, since shuffling data means shuffling the whole graph.
		callbacks=[es_callback, mc_callback],
	)
	sg.utils.plot_history(history)

	#--------------------
	# Evaluate the model.
	# Reload the saved weights of the best model found during the training.
	model.load_weights('logs/best_model.h5')

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
	emb_layer = next(l for l in model.layers if l.name.startswith('graph_attention'))
	print('Embedding layer: {}, output shape {}'.format(emb_layer.name, emb_layer.output_shape))

	embedding_model = tf.keras.Model(inputs=x_inp, outputs=emb_layer.output)
	emb = embedding_model.predict(all_gen)
	print(emb.shape)

	X = emb.squeeze(0)
	y = np.argmax(target_encoding.transform(node_subjects), axis=1)

	if X.shape[1] > 2:
		#transform = sklearn.decomposition.PCA
		transform = sklearn.manifold.TSNE

		trans = transform(n_components=2)
		emb_transformed = pd.DataFrame(trans.fit_transform(X), index=list(G.nodes()))
		emb_transformed['label'] = y
	else:
		emb_transformed = pd.DataFrame(X, index=list(G.nodes()))
		emb_transformed = emb_transformed.rename(columns={'0': 0, '1': 1})
		emb_transformed['label'] = y

	alpha = 0.7

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.scatter(
		emb_transformed[0],
		emb_transformed[1],
		c=emb_transformed['label'].astype('category'),
		cmap='jet',
		alpha=alpha,
	)
	ax.set(aspect='equal', xlabel='$X_1$', ylabel='$X_2$')
	plt.title('{} visualization of GAT embeddings for cora dataset'.format(transform.__name__))
	plt.show()

def main():
	# Graph classification.
	#supervised_graph_classification_with_GCN()

	# Link prediction.
	#link_prediction_with_GCN()

	# Node classification.
	#node_classification_with_GCN()
	node_classification_with_GAT()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
