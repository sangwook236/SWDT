#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [file] >> ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch_lightning/pl_graph_neural_network.py

import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch.nn import Linear
import torch_scatter
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt

# GCNConv.
class GCN(torch.nn.Module):
	def __init__(self):
		super(GCN, self).__init__()
		torch.manual_seed(12345)
		self.conv1 = GCNConv(165, 128)
		self.conv2 = GCNConv(128, 2)
		self.classifier = Linear(2, 1)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		h = self.conv1(x, edge_index)
		h = h.tanh()
		h = self.conv2(h, edge_index)
		embeddings = h.tanh()  # Final GNN embedding space.

		# Apply a final (linear) classifier.
		out = self.classifier(embeddings)

		# Return out, embeddings.
		return F.sigmoid(out)

# GATConv.
class GAT(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, args):
		super(GAT, self).__init__()
		self.args = args

		# Use our GAT message passing.
		self.conv1 = GATConv(input_dim, hidden_dim, heads=args['heads'])
		self.conv2 = GATConv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])

		self.post_mp = torch.nn.Sequential(
			torch.nn.Linear(args['heads'] * hidden_dim, hidden_dim),
			torch.nn.Dropout(args['dropout']),
			torch.nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		# Layer 1.
		x = self.conv1(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
		# Layer 2.
		x = self.conv2(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
		# MLP output.
		x = self.post_mp(x)
		return F.sigmoid(x)

# GATv2Conv.
class GATv2(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, args):
		super(GATv2, self).__init__()
		self.args = args

		# Use our gat message passing.
		self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=args['heads'])
		self.conv2 = GATv2Conv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])
		
		self.post_mp = torch.nn.Sequential(
			torch.nn.Linear(args['heads'] * hidden_dim, hidden_dim),
			torch.nn.Dropout(args['dropout']), 
			torch.nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		# Layer 1.
		x = self.conv1(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
		# Layer 2.
		x = self.conv2(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
		# MLP output.
		x = self.post_mp(x)
		return F.sigmoid(x)

# Custom GAT implementation.
class myGAT(torch_geometric.nn.conv.MessagePassing):
	def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.0, **kwargs):
		super(myGAT, self).__init__(node_dim=0, **kwargs)

		self.in_channels = in_channels  # Node features input dimension.
		self.out_channels = out_channels  # Node level output dimension.
		self.heads = heads  # No. of attention heads.
		self.negative_slope = negative_slope
		self.dropout = dropout

		self.lin_l = None
		self.lin_r = None
		self.att_l = None
		self.att_r = None

		# Initialization.
		self.lin_l = torch.nn.Linear(in_channels, heads * out_channels)
		self.lin_r = self.lin_l
		self.att_l = torch.nn.parameter.Parameter(torch.Tensor(1, heads, out_channels).float())
		self.att_r = torch.nn.parameter.Parameter(torch.Tensor(1, heads, out_channels).float())
		self.reset_parameters()

	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.lin_l.weight)
		torch.nn.init.xavier_uniform_(self.lin_r.weight)
		torch.nn.init.xavier_uniform_(self.att_l)
		torch.nn.init.xavier_uniform_(self.att_r)

	def forward(self, x, edge_index, size=None):
		H, C = self.heads, self.out_channels  # DIM：H, outC.

		# Linearly transform node feature matrix.
		x_source = self.lin_l(x).view(-1, H, C)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC].
		x_target = self.lin_r(x).view(-1, H, C)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC].

		# Alphas will be used to calculate attention later.
		alpha_l = (x_source * self.att_l).sum(dim=-1)  # DIM: [nodes, H, outC] x [H, outC] => [nodes, H].
		alpha_r = (x_target * self.att_r).sum(dim=-1)  # DIM: [nodes, H, outC] x [H, outC] => [nodes, H].

		# Start propagating messages (runs message and aggregate).
		out = self.propagate(edge_index, x=(x_source, x_target), alpha=(alpha_l, alpha_r), size=size)  # DIM: [nodes, H, outC].
		out = out.view(-1, self.heads * self.out_channels)  # DIM: [nodes, H * outC].

		return out

	def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
		# Calculate attention for edge pairs.
		attention = F.leaky_relu((alpha_j + alpha_i), self.negative_slope)  # EQ(1) DIM: [Edges, H].
		attention = torch_geometric.utils.softmax(attention, index, ptr, size_i)  # EQ(2) DIM: [Edges, H] | This softmax only calculates it over all neighbourhood nodes.
		attention = F.dropout(attention, p=self.dropout, training=self.training)  # DIM: [Edges, H].

		# Multiple attention with node features for all edges.
		out = x_j * attention.unsqueeze(-1)  # EQ(3.1) [Edges, H, outC] x [Edges, H] = [Edges, H, outC].

		return out

	def aggregate(self, inputs, index, dim_size=None):
		# EQ(3.2) For each node, aggregate messages for all neighbourhood nodes.
		out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')  # inputs (from message) DIM: [Edges, H, outC] => DIM: [Nodes, H, outC].
		return out

# Custom GATv2 model.
class myGATv2(torch_geometric.nn.conv.MessagePassing):
	def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0.0, **kwargs):
		super(myGATv2, self).__init__(node_dim=0, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.heads = heads
		self.negative_slope = negative_slope
		self.dropout = dropout

		self.lin_l = None
		self.lin_r = None
		self.att_l = None
		self.att_r = None
		self._alpha = None

		# self.lin_l is the linear transformation that you apply to embeddings BEFORE message passing.
		self.lin_l =  Linear(in_channels, heads * out_channels)
		self.lin_r = self.lin_l

		self.att = torch.nn.parameter.Parameter(torch.Tensor(1, heads, out_channels))
		self.reset_parameters()

	# Initialize parameters with xavier uniform.
	def reset_parameters(self):
		torch.nn.init.xavier_uniform_(self.lin_l.weight)
		torch.nn.init.xavier_uniform_(self.lin_r.weight)
		torch.nn.init.xavier_uniform_(self.att)

	def forward(self, x, edge_index, size=None):
		H, C = self.heads, self.out_channels  # DIM：H, outC.
		# Linearly transform node feature matrix.
		x_source = self.lin_l(x).view(-1, H, C)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC].
		x_target = self.lin_r(x).view(-1, H, C)  # DIM: [Nodex x In] [in x H * outC] => [nodes x H * outC] => [nodes, H, outC].

		# Start propagating messages (runs message and aggregate).
		out= self.propagate(edge_index, x=(x_source, x_target), size=size)  # DIM: [nodes, H, outC].
		out= out.view(-1, self.heads * self.out_channels)  # DIM: [nodes, H * outC].
		alpha = self._alpha
		self._alpha = None
		return out

	# Process a message passing.
	def message(self, x_j, x_i, index, ptr, size_i):
		# Computation using previous equations.
		x = x_i + x_j
		x = F.leaky_relu(x, self.negative_slope)  # See Equation above: Apply the non-linearty function.
		alpha = (x * self.att).sum(dim=-1)  # Apply attnention "a" layer after the non-linearity.
		alpha = torch_geometric.utils.softmax(alpha, index, ptr, size_i)  # This softmax only calculates it over all neighbourhood nodes.
		self._alpha = alpha
		alpha= F.dropout(alpha, p=self.dropout, training=self.training)
		# Multiple attention with node features for all edges.
		out = x_j * alpha.unsqueeze(-1)  

		return out

	# Aggregation of messages.
	def aggregate(self, inputs, index, dim_size=None):
		out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
		return out

# GATCustom.
class GATmodif(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, args):
		super(GATmodif, self).__init__()
		self.args = args

		# Use our gat message passing.
		self.conv1 = myGAT(input_dim, hidden_dim, heads=args['heads'])
		self.conv2 = myGAT(args['heads'] * hidden_dim, hidden_dim, heads=args['heads']) 

		self.post_mp = torch.nn.Sequential(
			torch.nn.Linear(args['heads'] * hidden_dim, hidden_dim),
			torch.nn.Dropout(args['dropout']),
			torch.nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		# Layer 1.
		x = self.conv1(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# Layer 2.
		x = self.conv2(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# MLP output.
		x = self.post_mp(x)
		return F.sigmoid(x)

class GATv2modif(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, args):
		super(GATv2modif, self).__init__()
		self.args = args

		# Use our gat message passing.
		self.conv1 = myGATv2(input_dim, hidden_dim, heads=args['heads']) 
		self.conv2 = myGATv2(args['heads'] *hidden_dim, hidden_dim, heads=args['heads'])

		self.post_mp = torch.nn.Sequential(
			torch.nn.Linear(args['heads'] * hidden_dim, hidden_dim),
			torch.nn.Dropout(args['dropout']),
			torch.nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		# Layer 1.
		x = self.conv1(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# Layer 2.
		x = self.conv2(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# MLP output.
		x = self.post_mp(x)
		return F.sigmoid(x)

class GATmodif_3layer(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, args):
		super(GATmodif_3layer, self).__init__()
		self.args = args

		# Use our gat message passing.
		self.conv1 = myGAT(input_dim, hidden_dim, heads=args['heads'])
		self.conv2 = myGAT(args['heads'] * hidden_dim, hidden_dim, heads=args['heads']) 
		self.conv3 = myGAT(args['heads'] * hidden_dim, hidden_dim, heads=args['heads']) 

		self.post_mp = torch.nn.Sequential(
			torch.nn.Linear(args['heads'] * hidden_dim, hidden_dim),
			torch.nn.Dropout(args['dropout']), 
			torch.nn.Linear(hidden_dim, output_dim)
		)

	def forward(self, data, adj=None):
		x, edge_index = data.x, data.edge_index
		# Layer 1.
		x = self.conv1(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# Layer 2.
		x = self.conv2(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)
		
		# Layer 3.
		x = self.conv3(x, edge_index)
		x = F.dropout(F.relu(x), p=self.args['dropout'], training=self.training)

		# MLP output.
		x = self.post_mp(x)
		return F.sigmoid(x)

# Metric manager.
class MetricManager(object):
	def __init__(self, modes=["train", "val"]):
		self.output = {}

		for mode in modes:
			self.output[mode] = {}
			self.output[mode]["accuracy"] = []
			self.output[mode]["f1micro"] = []
			self.output[mode]["f1macro"] = []
			self.output[mode]["aucroc"] = []
			# New.
			self.output[mode]["precision"] = []
			self.output[mode]["recall"] = []
			self.output[mode]["cm"] = []

	def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):
		# Calculate metrics.
		pred_labels = pred_scores > threshold
		accuracy = accuracy_score(target_labels, pred_labels)
		f1micro = f1_score(target_labels, pred_labels, average='micro')
		f1macro = f1_score(target_labels, pred_labels, average='macro')
		aucroc = roc_auc_score(target_labels, pred_scores)
		# New.
		recall = recall_score(target_labels, pred_labels)
		precision = precision_score(target_labels, pred_labels)
		cm = confusion_matrix(target_labels, pred_labels)

		# Collect results.
		self.output[mode]["accuracy"].append(accuracy)
		self.output[mode]["f1micro"].append(f1micro)
		self.output[mode]["f1macro"].append(f1macro)
		self.output[mode]["aucroc"].append(aucroc)
		# New.
		self.output[mode]["recall"].append(recall)
		self.output[mode]["precision"].append(precision)
		self.output[mode]["cm"].append(cm)

		return accuracy, f1micro,f1macro, aucroc,recall,precision,cm

	# Get best results.
	def get_best(self, metric, mode="val"):
		# Get best results index.
		best_results = {}
		i = np.array(self.output[mode][metric]).argmax()

		# Output.
		for m in self.output[mode].keys():
			best_results[m] = self.output[mode][m][i]

		return best_results

# REF [site] >> https://colab.research.google.com/drive/1N5yiB10Zbk84kA4H-Pt1kizNmPmwyx3A?usp=sharing#scrollTo=LMM7lBiU2I6F
def fraud_detection_with_graph_attention_networks():
	# Elliptic Data Set - Bitcoin Transaction Graph:
	#	https://www.kaggle.com/ellipticco/elliptic-data-set
	#
	#	https://drive.google.com/uc?id=1CIFpAquzYBA98gQCdMb92fC0w6yrYf2m
	#	https://drive.google.com/uc?id=1Cfh0VIXWTc8EK96WRZdyaqgA2-JwvsUG
	#	https://drive.google.com/uc?id=1Cfh8hA9Tl8uCPrLSmcIQI3qCbEjOFl7C

	data_dir_path = "."
	save_dir_path = data_dir_path + "/save_results"

	os.makedirs(save_dir_path, exist_ok=False)

	# Load data from the folder.
	df_features = pd.read_csv(data_dir_path + "/elliptic_txs_features.csv", header=None)
	df_edges = pd.read_csv(data_dir_path + "/elliptic_txs_edgelist.csv")
	df_classes =  pd.read_csv(data_dir_path + "/elliptic_txs_classes.csv")

	# Add class names for easy understanding.
	# Reformat classes 0: licit, 1: illicit, 2: unknown.
	df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0}) 

	# See repartition of nodes per class.
	group_class = df_classes.groupby('class').count()
	plt.title("# of nodes per class")
	plt.barh(['Licit', 'Illicit', 'Unknown'], group_class['txId'].values, color=['g', 'orange', 'r'])

	plt.show()

	# View node features.
	# Data is each node is a transaction ID, and edges are a bit weird, but its like a "previous transaction" edge type
	# colume 0 = transaction id
	print(df_features.head())

	# See edges.
	print(df_edges.head())

	# See classes.
	print(df_classes.head())

	# Merge features with classes.
	df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
	df_merge = df_merge.sort_values(0).reset_index(drop=True)
	print(df_merge.head())

	#--------------------
	# Edge index: Map trans IDs to node IDs.

	# Setup trans ID to node ID mapping.
	nodes = df_merge[0].values

	map_id = {j:i for i,j in enumerate(nodes)}  # Mapping nodes to indexes.

	# Create edge df that has transID mapped to nodeIDs.
	edges = df_edges.copy()
	edges.txId1 = edges.txId1.map(map_id)  # Get nodes idx1 from edges list and filtered data.
	edges.txId2 = edges.txId2.map(map_id)

	edges = edges.astype(int)

	edge_index = np.array(edges.values).T  # Convert into an array.
	edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()  # Create a tensor.

	print("Shape of edge index is {}".format(edge_index.shape))
	print(edge_index)

	# Create weights tensor with same shape of edge_index.
	weights = torch.tensor([1] * edge_index.shape[1] , dtype=torch.double) 

	# Define labels.
	labels = df_merge['class'].values
	print("Lables: {}".format(np.unique(labels)))
	print(labels)

	#--------------------
	# Node Features.

	# Mapping txIds to corresponding indices, to pass node features to the model.
	node_features = df_merge.drop(['txId'], axis=1).copy()
	#node_features[0] = node_features[0].map(map_id)  # Convert transaction ID to node ID.
	print("Unique = {}".format(node_features["class"].unique()))

	# Retain known vs unknown IDs.
	classified_idx = node_features['class'].loc[node_features['class'] != 2].index   # Filter on known labels.
	unclassified_idx = node_features['class'].loc[node_features['class'] == 2].index

	classified_illicit_idx = node_features['class'].loc[node_features['class'] == 1].index  # Filter on illicit labels.
	classified_licit_idx = node_features['class'].loc[node_features['class'] == 0].index  # Filter on licit labels.

	# Drop unwanted columns, 0 = transID, 1 = time period, class = labels.
	node_features = node_features.drop(columns=[0, 1, 'class'])

	# Convert to tensor.
	node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double), dtype=torch.double)  # Drop unused columns.
	print(node_features_t)

	# See node features again.
	print(node_features)

	#--------------------

	# Train test splits.
	# Create a known vs unknown mask.
	train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.15)
	print("train_idx size {}".format(len(train_idx)))
	print("tets_idx size {}".format(len(valid_idx)))

	# Create a PyG dataset.
	data_train = Data(x=node_features_t, edge_index=edge_index, edge_attr=weights, y=torch.tensor(labels, dtype=torch.double))
	# Add in the train and valid idx.
	data_train.train_idx = train_idx
	data_train.valid_idx = valid_idx
	data_train.test_idx = unclassified_idx

	#--------------------
	# Model training.

	# GNNTrainer object.
	class GnnTrainer(object):
		def __init__(self, model):
			self.model = model
			self.metric_manager = MetricManager(modes=["train", "val"])

		def train(self, data_train, optimizer, criterion, scheduler, args):
			self.data_train = data_train
			for epoch in range(args['epochs']):
				self.model.train()
				optimizer.zero_grad()
				out = self.model(data_train)

				out = out.reshape((data_train.x.shape[0]))
				loss = criterion(out[data_train.train_idx], data_train.y[data_train.train_idx])
				# Metric calculations.
				# Train data.
				target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
				pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
				train_acc, train_f1, train_f1macro, train_aucroc, train_recall, train_precision, train_cm = self.metric_manager.store_metrics("train", pred_scores, target_labels)

				# Training step.
				loss.backward()
				optimizer.step()

				# Validation data.
				self.model.eval()
				target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
				pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
				val_acc, val_f1, val_f1macro, val_aucroc, val_recall, val_precision, val_cm = self.metric_manager.store_metrics("val", pred_scores, target_labels)

				if epoch % 5 == 0:
					print("epoch: {} - loss: {:.4f} - accuracy train: {:.4f} - accuracy val: {:.4f} - roc val: {:.4f} - f1micro val: {:.4f}".format(epoch, loss.item(), train_acc, val_acc, val_aucroc, val_f1))

		# Predict labels.
		def predict(self, data=None, unclassified_only=True, threshold=0.5):
			# Evaluate model.
			self.model.eval()
			if data is not None:
				self.data_train = data

			out = self.model(self.data_train)
			out = out.reshape((self.data_train.x.shape[0]))

			if unclassified_only:
				pred_scores = out.detach().cpu().numpy()[self.data_train.test_idx]
			else:
				pred_scores = out.detach().cpu().numpy()

			pred_labels = pred_scores > threshold

			return {"pred_scores": pred_scores, "pred_labels": pred_labels}

		# Save metrics.
		def save_metrics(self, save_name):
			file_to_store = open(save_name, "wb")
			pickle.dump(self.metric_manager, file_to_store)
			file_to_store.close()
		
		# Save model.
		def save_model(self, save_name):
			torch.save(self.model.state_dict(), save_name)

	# Training and validation.

	# Set training arguments, set prebuild=True to use builtin PyG models otherwise False.
	args = {
		'epochs': 100,
		'lr': 0.01,
		'weight_decay': 1e-5,
		'prebuild': True,
		'heads': 2,
		'hidden_dim': 128, 
		'dropout': 0.5
	}

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Push data to GPU.
	data_train = data_train.to(device)

	#-----
	"""
	# Model selector GAT or GATv2.
	net = "GAT"

	if net == "GAT":
		if args['prebuild'] == True:
			model = GAT(data_train.num_node_features, args['hidden_dim'], 1, args)
			print("Prebuilt GAT from PyG ")
		else:
			model = GATmodif(data_train.num_node_features, args['hidden_dim'], 1, args)
			print("Custom GAT implemented")
	elif net == "GATv2":
		#args['heads'] = 1
		if args['prebuild'] == True:
			model = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args) 
			print("Prebuilt GATv2 from PyG ")
		else:
			model = GATv2modif(data_train.num_node_features, args['hidden_dim'], 1, args) 
			print("Custom GATv2 implemented")
	"""

	# Here we run GAT model.
	print("Prebuilt GAT from PyG ")
	model = GAT(data_train.num_node_features, args['hidden_dim'], 1, args)
	model.double().to(device)

	# Setup training settings.
	optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	criterion = torch.nn.BCELoss()

	# Train.
	gnn_trainer_gat = GnnTrainer(model)
	gnn_trainer_gat.train(data_train, optimizer, criterion, scheduler, args)

	gnn_trainer_gat.save_metrics(save_dir_path + "/GATprebuilt.results")
	gnn_trainer_gat.save_model(save_dir_path + "/GATprebuilt.pth")

	#-----
	# Here we run GATv2 model.
	print("Prebuilt GATv2 from PyG ")
	model = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args)
	model.double().to(device)

	# Setup training settings.
	optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	criterion = torch.nn.BCELoss()

	# Train.
	gnn_trainer_gatv2 = GnnTrainer(model)
	gnn_trainer_gatv2.train(data_train, optimizer, criterion, scheduler, args)

	gnn_trainer_gatv2.save_metrics(save_dir_path + "/GATv2prebuilt.results")
	gnn_trainer_gatv2.save_model(save_dir_path + "/GATv2prebuilt.pth")

	#-----
	# Here we run GATmodif model.
	print("Custom GAT implemented")
	model = GATmodif(data_train.num_node_features, args['hidden_dim'], 1, args)
	model.double().to(device)

	# Setup training settings.
	optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	criterion = torch.nn.BCELoss()

	# Train.
	gnn_trainer_gatmodif = GnnTrainer(model)
	gnn_trainer_gatmodif.train(data_train, optimizer, criterion, scheduler, args)

	gnn_trainer_gatmodif.save_metrics(save_dir_path + "/GATcustom.results")
	gnn_trainer_gatmodif.save_model(save_dir_path + "/GATcustom.pth")

	#-----
	# Here we run GATmodif_3layer model.
	print("Custom GAT 3layer implemented")
	model = GATmodif_3layer(data_train.num_node_features, args['hidden_dim'], 1, args)
	model.double().to(device)

	# Setup training settings.
	optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	criterion = torch.nn.BCELoss()

	# Train.
	gnn_trainer_gat3layer = GnnTrainer(model)
	gnn_trainer_gat3layer.train(data_train, optimizer, criterion, scheduler, args)

	gnn_trainer_gat3layer.save_metrics(save_dir_path + "/GAT3layers.results")
	gnn_trainer_gat3layer.save_model(save_dir_path + "/GAT3layers.pth")

	#-----
	# Here we run GATv2modif model.
	print("Custom GATv2 implemented")
	model = GATv2modif(data_train.num_node_features, args['hidden_dim'], 1, args)
	model.double().to(device)

	# Setup training settings.
	optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
	criterion = torch.nn.BCELoss()

	# Train.
	gnn_trainer_gatv2modif = GnnTrainer(model)
	gnn_trainer_gatv2modif.train(data_train, optimizer, criterion, scheduler, args)

	gnn_trainer_gatv2modif.save_metrics(save_dir_path + "/GATv2custom.results")
	gnn_trainer_gatv2modif.save_model(save_dir_path + "/GATv2custom.pth")

	# Fetch results from saved.
	mmGATprebuilt = pickle.load(open(save_dir_path + "/GATprebuilt.results", "rb"))
	print(mmGATprebuilt.get_best("aucroc"))

	mmGATv2prebuilt = pickle.load(open(save_dir_path + "/GATv2prebuilt.results", "rb"))
	print(mmGATv2prebuilt.get_best("aucroc"))

	#--------------------
	# Performance visualization.

	# Validation accuracy comparisons.
	def plot_training_comp(metric_manager_list, names, metric="aucroc", version="val", title="Val set accuracy comparison"):
		import plotly.graph_objects as go

		fig = go.Figure()
		fig = fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title=metric)

		for i, metric_manager in enumerate(metric_manager_list):
			epochs = np.arange(len(metric_manager.output[version][metric]))
			fig.add_trace(go.Scatter(x=epochs, y=metric_manager.output[version][metric], name=names[i]))
		fig.show()

	# Load saved outputs.
	def load_results(save_path):
		mm1 = pickle.load(open(save_path, "rb"))
		return mm1

	# Compare GAT vs GATv2 with 2 heads, prebuilt PyGs models.
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "f1micro", title="Val set f1micro, heads=2")
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "accuracy", title="Val set accuracy, heads=2")
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "f1macro", title="Val set f1macro, heads=2")
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "aucroc", title="Val set aucroc, heads=2")
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "recall", title="Val set recall, heads=2")
	plot_training_comp([mmGATprebuilt, mmGATv2prebuilt], ["GAT", "GATv2" ], "precision", title="Val set precision, heads=2")

	# Graph visualization.

	# Load one model.
	m1 = GATv2(data_train.num_node_features, args['hidden_dim'], 1, args).to(device).double()
	m1.load_state_dict(torch.load(save_dir_path + "/GATv2prebuilt.pth"))
	gnn_t2 = GnnTrainer(m1)
	output = gnn_t2.predict(data=data_train, unclassified_only=False)
	print(output)

	# Get index for one time period.
	time_period = 28
	sub_node_list = df_merge.index[df_merge.loc[:,1] == time_period].tolist()

	# Fetch list of edges for that time period.
	edge_tuples = []
	for row in data_train.edge_index.view(-1, 2).numpy():
		if (row[0] in sub_node_list) | (row[1] in sub_node_list):
			edge_tuples.append(tuple(row))
	print(len(edge_tuples))

	# Fetch predicted results for that time period.
	node_color = []
	for node_id in sub_node_list:
		if node_id in classified_illicit_idx:
			label = "red"  # Fraud.
		elif node_id in classified_licit_idx:
			label = "green"  # Not fraud.
		else:
			if output['pred_labels'][node_id]:
				label = "orange"  # Predicted fraud.
			else:
				label = "blue"  # Not fraud predicted.

		node_color.append(label)

	# Setup networkx graph.
	G = nx.Graph()
	G.add_edges_from(edge_tuples)

	# Plot the graph.
	plt.figure(3, figsize=(16, 16)) 
	plt.title("Time period: " + str(time_period))
	nx.draw_networkx(G, nodelist=sub_node_list, node_color=node_color, node_size=6, with_labels=False)

	# Training Cross comparisons.

	# Take names of the saved files.
	#model_name_list = [save_dir_path + "/GATv2prebuilt.results", save_dir_path + "/GATprebuilt.results"]
	model_name_list = [save_dir_path + "/GATv2prebuilt.results", save_dir_path + "/GATprebuilt.results", save_dir_path + "/GATv2custom.results", save_dir_path + "/GATcustom.results", save_dir_path + "/GAT3layers.results"]
	# Assign names to plots.
	#names = ["GATv2_prebuilt", "GAT_prebuilt"]
	names = ["GATv2_prebuilt", "GAT_prebuilt", "GATv2custom", "GATcustom", "GAT3layers"]

	# Iterate to load saved outputs and plots.
	mm_list0 = [load_results(path) for path in model_name_list]
	plot_training_comp(mm_list0, names, "aucroc", title="Val set aucroc comparison")

	# Best metrics.

	# Plot best results of each into a table.
	aucroc_l = []
	accuracy_l = []
	f1micro_l = []
	f1macro_l = []
	model_versions = []
	for c1, mm in enumerate(mm_list0):
		model_versions.append(names[c1])
		best = mm.get_best("aucroc")

		aucroc_l.append(best["aucroc"])
		accuracy_l.append(best["accuracy"])
		f1micro_l.append(best["f1micro"])
		f1macro_l.append(best["f1macro"])

	d = {"model_version": model_versions, "aucroc": aucroc_l, "accuracy": accuracy_l, "f1_macro": f1macro_l, "f1_micro": f1micro_l}
	df = pd.DataFrame(data=d)

	print(df.round(3))

def main():
	fraud_detection_with_graph_attention_networks()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
