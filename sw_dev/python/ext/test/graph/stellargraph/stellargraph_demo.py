#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt
#%matplotlib inline

# REF [site] >> https://stellargraph.readthedocs.io/en/stable/demos/node-classification/gcn-node-classification.html
def classify_nodes_with_GCN():
	# Load the CORA network.
	dataset = sg.datasets.Cora()
	display(dataset.description)

	G, node_subjects = dataset.load()
	print(G.info())
	print(node_subjects.value_counts().to_frame())

	# Split the data.
	train_subjects, test_subjects = model_selection.train_test_split(
		node_subjects, train_size=140, test_size=None, stratify=node_subjects
	)
	val_subjects, test_subjects = model_selection.train_test_split(
		test_subjects, train_size=500, test_size=None, stratify=test_subjects
	)
	print(train_subjects.value_counts().to_frame())

	# Convert to numeric arrays.
	target_encoding = preprocessing.LabelBinarizer()

	train_targets = target_encoding.fit_transform(train_subjects)
	val_targets = target_encoding.transform(val_subjects)
	test_targets = target_encoding.transform(test_subjects)

	#--------------------
	# Create the GCN layers.
	generator = FullBatchNodeGenerator(G, method='gcn')
	gcn = GCN(layer_sizes=[16, 16], activations=['relu', 'relu'], generator=generator, dropout=0.5)
	x_inp, x_out = gcn.in_out_tensors()

	predictions = layers.Dense(units=train_targets.shape[1], activation='softmax')(x_out)

	#--------------------
	# Train the model.
	model = Model(inputs=x_inp, outputs=predictions)
	model.compile(
		optimizer=optimizers.Adam(lr=0.01),
		loss=losses.categorical_crossentropy,
		metrics=['acc'],
	)

	train_gen = generator.flow(train_subjects.index, train_targets)
	val_gen = generator.flow(val_subjects.index, val_targets)

	from tensorflow.keras.callbacks import EarlyStopping
	es_callback = EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True)

	history = model.fit(
		train_gen,
		epochs=200,
		validation_data=val_gen,
		verbose=2,
		shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
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
	embedding_model = Model(inputs=x_inp, outputs=x_out)
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
	classify_nodes_with_GCN()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
