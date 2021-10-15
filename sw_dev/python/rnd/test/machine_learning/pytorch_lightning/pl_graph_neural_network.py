#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import urllib.request
from urllib.error import HTTPError
import torch, torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/06-graph-neural-networks.html
def simple_gnn_tutorial():
	AVAIL_GPUS = min(1, torch.cuda.device_count())
	BATCH_SIZE = 256 if AVAIL_GPUS else 64
	# Path to the folder where the datasets are/should be downloaded.
	DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
	# Path to the folder where the pretrained models are saved.
	CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

	# Setting the seed.
	pl.seed_everything(42)

	# Ensure that all operations are deterministic on GPU (if used) for reproducibility.
	torch.backends.cudnn.determinstic = True
	torch.backends.cudnn.benchmark = False

	# Github URL where saved models are stored for this tutorial.
	base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
	# Files to download.
	pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]

	# Create checkpoint path if it doesn't exist yet.
	os.makedirs(CHECKPOINT_PATH, exist_ok=True)

	# For each file, check whether it already exists. If not, try downloading it.
	for file_name in pretrained_files:
		file_path = os.path.join(CHECKPOINT_PATH, file_name)
		if "/" in file_name:
			os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
		if not os.path.isfile(file_path):
			file_url = base_url + file_name
			print("Downloading %s..." % file_url)
			try:
				urllib.request.urlretrieve(file_url, file_path)
			except HTTPError as e:
				print(
					"Something went wrong. Please try to download the file from the GDrive folder,"
					" or contact the author with the full output including the following error:\n",
					e,
				)

	#--------------------
	# Graph convolutions.

	class GCNLayer(nn.Module):
		def __init__(self, c_in, c_out):
			super().__init__()
			self.projection = nn.Linear(c_in, c_out)

		def forward(self, node_feats, adj_matrix):
			"""
			Args:
				node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
				adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
							adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
							Assumes to already have added the identity connections.
							Shape: [batch_size, num_nodes, num_nodes]
			"""
			# Num neighbours = number of incoming edges.
			num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
			node_feats = self.projection(node_feats)
			node_feats = torch.bmm(adj_matrix, node_feats)
			node_feats = node_feats / num_neighbours
			return node_feats

	node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
	adj_matrix = torch.Tensor([[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]])

	print("Node features:\n", node_feats)
	print("\nAdjacency matrix:\n", adj_matrix)

	layer = GCNLayer(c_in=2, c_out=2)
	layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
	layer.projection.bias.data = torch.Tensor([0.0, 0.0])

	with torch.no_grad():
		out_feats = layer(node_feats, adj_matrix)

	print("Adjacency matrix", adj_matrix)
	print("Input features", node_feats)
	print("Output features", out_feats)

	#--------------------
	# Graph attention.

	class GATLayer(nn.Module):
		def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
			"""
			Args:
				c_in: Dimensionality of input features
				c_out: Dimensionality of output features
				num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
							output features are equally split up over the heads if concat_heads=True.
				concat_heads: If True, the output of the different heads is concatenated instead of averaged.
				alpha: Negative slope of the LeakyReLU activation.
			"""
			super().__init__()
			self.num_heads = num_heads
			self.concat_heads = concat_heads
			if self.concat_heads:
				assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
				c_out = c_out // num_heads

			# Sub-modules and parameters needed in the layer.
			self.projection = nn.Linear(c_in, c_out * num_heads)
			self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head.
			self.leakyrelu = nn.LeakyReLU(alpha)

			# Initialization from the original implementation.
			nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
			nn.init.xavier_uniform_(self.a.data, gain=1.414)

		def forward(self, node_feats, adj_matrix, print_attn_probs=False):
			"""
			Args:
				node_feats: Input features of the node. Shape: [batch_size, c_in]
				adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
				print_attn_probs: If True, the attention weights are printed during the forward pass
								(for debugging purposes)
			"""
			batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

			# Apply linear layer and sort nodes by head.
			node_feats = self.projection(node_feats)
			node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

			# We need to calculate the attention logits for every edge in the adjacency matrix.
			# Doing this on all possible combinations of nodes is very expensive
			# => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges.
			# Returns indices where the adjacency matrix is not 0 => edges.
			edges = adj_matrix.nonzero(as_tuple=False)
			node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
			edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
			edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
			a_input = torch.cat(
				[
					torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
					torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
				],
				dim=-1,
			)  # Index select returns a tensor with node_feats_flat being indexed at the desired positions.

			# Calculate attention MLP output (independent for each head).
			attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
			attn_logits = self.leakyrelu(attn_logits)

			# Map list of attention values back into a matrix.
			attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
			attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

			# Weighted average of attention.
			attn_probs = F.softmax(attn_matrix, dim=2)
			if print_attn_probs:
				print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
			node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

			# If heads should be concatenated, we can do this by reshaping. Otherwise, take mean.
			if self.concat_heads:
				node_feats = node_feats.reshape(batch_size, num_nodes, -1)
			else:
				node_feats = node_feats.mean(dim=2)

			return node_feats

	layer = GATLayer(2, 2, num_heads=2)
	layer.projection.weight.data = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
	layer.projection.bias.data = torch.Tensor([0.0, 0.0])
	layer.a.data = torch.Tensor([[-0.2, 0.3], [0.1, -0.1]])

	with torch.no_grad():
		out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

	print("Adjacency matrix", adj_matrix)
	print("Input features", node_feats)
	print("Output features", out_feats)

	#--------------------
	# PyTorch Geometric.

	gnn_layer_by_name = {"GCN": geom_nn.GCNConv, "GAT": geom_nn.GATConv, "GraphConv": geom_nn.GraphConv}

	#--------------------
	# Node-level tasks: Semi-supervised node classification.

	cora_dataset = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
	print(cora_dataset[0])

	class GNNModel(nn.Module):
		def __init__(
			self,
			c_in,
			c_hidden,
			c_out,
			num_layers=2,
			layer_name="GCN",
			dp_rate=0.1,
			**kwargs,
		):
			"""
			Args:
				c_in: Dimension of input features
				c_hidden: Dimension of hidden features
				c_out: Dimension of the output features. Usually number of classes in classification
				num_layers: Number of "hidden" graph layers
				layer_name: String of the graph layer to use
				dp_rate: Dropout rate to apply throughout the network
				kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
			"""
			super().__init__()
			gnn_layer = gnn_layer_by_name[layer_name]

			layers = []
			in_channels, out_channels = c_in, c_hidden
			for l_idx in range(num_layers - 1):
				layers += [
					gnn_layer(in_channels=in_channels, out_channels=out_channels, **kwargs),
					nn.ReLU(inplace=True),
					nn.Dropout(dp_rate),
				]
				in_channels = c_hidden
			layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, **kwargs)]
			self.layers = nn.ModuleList(layers)

		def forward(self, x, edge_index):
			"""
			Args:
				x: Input features per node
				edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
			"""
			for layer in self.layers:
				# For graph layers, we need to add the "edge_index" tensor as additional input
				# All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
				# we can simply check the class type.
				if isinstance(layer, geom_nn.MessagePassing):
					x = layer(x, edge_index)
				else:
					x = layer(x)
			return x

	class MLPModel(nn.Module):
		def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
			"""
			Args:
				c_in: Dimension of input features
				c_hidden: Dimension of hidden features
				c_out: Dimension of the output features. Usually number of classes in classification
				num_layers: Number of hidden layers
				dp_rate: Dropout rate to apply throughout the network
			"""
			super().__init__()
			layers = []
			in_channels, out_channels = c_in, c_hidden
			for l_idx in range(num_layers - 1):
				layers += [nn.Linear(in_channels, out_channels), nn.ReLU(inplace=True), nn.Dropout(dp_rate)]
				in_channels = c_hidden
			layers += [nn.Linear(in_channels, c_out)]
			self.layers = nn.Sequential(*layers)

		def forward(self, x, *args, **kwargs):
			"""
			Args:
				x: Input features per node
			"""
			return self.layers(x)

	class NodeLevelGNN(pl.LightningModule):
		def __init__(self, model_name, **model_kwargs):
			super().__init__()
			# Saving hyperparameters.
			self.save_hyperparameters()

			if model_name == "MLP":
				self.model = MLPModel(**model_kwargs)
			else:
				self.model = GNNModel(**model_kwargs)
			self.loss_module = nn.CrossEntropyLoss()

		def forward(self, data, mode="train"):
			x, edge_index = data.x, data.edge_index
			x = self.model(x, edge_index)

			# Only calculate the loss on the nodes corresponding to the mask.
			if mode == "train":
				mask = data.train_mask
			elif mode == "val":
				mask = data.val_mask
			elif mode == "test":
				mask = data.test_mask
			else:
				assert False, "Unknown forward mode: %s" % mode

			loss = self.loss_module(x[mask], data.y[mask])
			acc = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
			return loss, acc

		def configure_optimizers(self):
			# We use SGD here, but Adam works as well.
			optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-3)
			return optimizer

		def training_step(self, batch, batch_idx):
			loss, acc = self.forward(batch, mode="train")
			self.log("train_loss", loss)
			self.log("train_acc", acc)
			return loss

		def validation_step(self, batch, batch_idx):
			_, acc = self.forward(batch, mode="val")
			self.log("val_acc", acc)

		def test_step(self, batch, batch_idx):
			_, acc = self.forward(batch, mode="test")
			self.log("test_acc", acc)

	def train_node_classifier(model_name, dataset, **model_kwargs):
		pl.seed_everything(42)
		node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

		# Create a PyTorch Lightning trainer.
		root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
		os.makedirs(root_dir, exist_ok=True)
		trainer = pl.Trainer(
			default_root_dir=root_dir,
			callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
			gpus=AVAIL_GPUS,
			max_epochs=200,
			progress_bar_refresh_rate=0,
		)  # 0 because epoch size is 1.
		trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need.

		# Check whether pretrained model exists. If yes, load it and skip training.
		pretrained_filename = os.path.join(CHECKPOINT_PATH, "NodeLevel%s.ckpt" % model_name)
		if os.path.isfile(pretrained_filename):
			print("Found pretrained model, loading...")
			model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
		else:
			pl.seed_everything()
			model = NodeLevelGNN(
				model_name=model_name, c_in=dataset.num_node_features, c_out=dataset.num_classes, **model_kwargs
			)
			trainer.fit(model, node_data_loader, node_data_loader)
			model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

		# Test best model on the test set.
		test_result = trainer.test(model, test_dataloaders=node_data_loader, verbose=False)
		batch = next(iter(node_data_loader))
		batch = batch.to(model.device)
		_, train_acc = model.forward(batch, mode="train")
		_, val_acc = model.forward(batch, mode="val")
		result = {"train": train_acc, "val": val_acc, "test": test_result[0]["test_acc"]}
		return model, result

	# Small function for printing the test scores.
	def print_results(result_dict):
		if "train" in result_dict:
			print("Train accuracy: %4.2f%%" % (100.0 * result_dict["train"]))
		if "val" in result_dict:
			print("Val accuracy:   %4.2f%%" % (100.0 * result_dict["val"]))
		print("Test accuracy:  %4.2f%%" % (100.0 * result_dict["test"]))

	node_mlp_model, node_mlp_result = train_node_classifier(
		model_name="MLP", dataset=cora_dataset, c_hidden=16, num_layers=2, dp_rate=0.1
	)

	print_results(node_mlp_result)

	node_gnn_model, node_gnn_result = train_node_classifier(
		model_name="GNN", layer_name="GCN", dataset=cora_dataset, c_hidden=16, num_layers=2, dp_rate=0.1
	)
	print_results(node_gnn_result)

	#--------------------
	# Edge-level tasks: Link prediction.

	#--------------------
	# Graph-level tasks: Graph classification.

	tu_dataset = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG")

	print("Data object:", tu_dataset.data)
	print("Length:", len(tu_dataset))
	print("Average label: %4.2f" % (tu_dataset.data.y.float().mean().item()))

	torch.manual_seed(42)
	tu_dataset.shuffle()
	train_dataset = tu_dataset[:150]
	test_dataset = tu_dataset[150:]

	graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=BATCH_SIZE)  # Additional loader for a larger datasets.
	graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

	graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
	graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=BATCH_SIZE)  # Additional loader for a larger datasets.
	graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

	batch = next(iter(graph_test_loader))
	print("Batch:", batch)
	print("Labels:", batch.y[:10])
	print("Batch indices:", batch.batch[:40])

	class GraphGNNModel(nn.Module):
		def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
			"""
			Args:
				c_in: Dimension of input features
				c_hidden: Dimension of hidden features
				c_out: Dimension of output features (usually number of classes)
				dp_rate_linear: Dropout rate before the linear layer (usually much higher than inside the GNN)
				kwargs: Additional arguments for the GNNModel object
			"""
			super().__init__()
			self.GNN = GNNModel(c_in=c_in, c_hidden=c_hidden, c_out=c_hidden, **kwargs)  # Not our prediction output yet!
			self.head = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out))

		def forward(self, x, edge_index, batch_idx):
			"""
			Args:
				x: Input features per node
				edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
				batch_idx: Index of batch element for each node
			"""
			x = self.GNN(x, edge_index)
			x = geom_nn.global_mean_pool(x, batch_idx)  # Average pooling.
			x = self.head(x)
			return x

	class GraphLevelGNN(pl.LightningModule):
		def __init__(self, **model_kwargs):
			super().__init__()
			# Saving hyperparameters.
			self.save_hyperparameters()

			self.model = GraphGNNModel(**model_kwargs)
			self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

		def forward(self, data, mode="train"):
			x, edge_index, batch_idx = data.x, data.edge_index, data.batch
			x = self.model(x, edge_index, batch_idx)
			x = x.squeeze(dim=-1)

			if self.hparams.c_out == 1:
				preds = (x > 0).float()
				data.y = data.y.float()
			else:
				preds = x.argmax(dim=-1)
			loss = self.loss_module(x, data.y)
			acc = (preds == data.y).sum().float() / preds.shape[0]
			return loss, acc

		def configure_optimizers(self):
			# High lr because of small dataset and small model.
			optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0)
			return optimizer

		def training_step(self, batch, batch_idx):
			loss, acc = self.forward(batch, mode="train")
			self.log("train_loss", loss)
			self.log("train_acc", acc)
			return loss

		def validation_step(self, batch, batch_idx):
			_, acc = self.forward(batch, mode="val")
			self.log("val_acc", acc)

		def test_step(self, batch, batch_idx):
			_, acc = self.forward(batch, mode="test")
			self.log("test_acc", acc)

	def train_graph_classifier(model_name, **model_kwargs):
		pl.seed_everything(42)

		# Create a PyTorch Lightning trainer with the generation callback.
		root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
		os.makedirs(root_dir, exist_ok=True)
		trainer = pl.Trainer(
			default_root_dir=root_dir,
			callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
			gpus=AVAIL_GPUS,
			max_epochs=500,
			progress_bar_refresh_rate=0,
		)
		trainer.logger._default_hp_metric = None

		# Check whether pretrained model exists. If yes, load it and skip training.
		pretrained_filename = os.path.join(CHECKPOINT_PATH, "GraphLevel%s.ckpt" % model_name)
		if os.path.isfile(pretrained_filename):
			print("Found pretrained model, loading...")
			model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
		else:
			pl.seed_everything(42)
			model = GraphLevelGNN(
				c_in=tu_dataset.num_node_features,
				c_out=1 if tu_dataset.num_classes == 2 else tu_dataset.num_classes,
				**model_kwargs,
			)
			trainer.fit(model, graph_train_loader, graph_val_loader)
			model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

		# Test best model on validation and test set.
		train_result = trainer.test(model, test_dataloaders=graph_train_loader, verbose=False)
		test_result = trainer.test(model, test_dataloaders=graph_test_loader, verbose=False)
		result = {"test": test_result[0]["test_acc"], "train": train_result[0]["test_acc"]}
		return model, result

	model, result = train_graph_classifier(
		model_name="GraphConv", c_hidden=256, layer_name="GraphConv", num_layers=3, dp_rate_linear=0.5, dp_rate=0.0
	)

	print("Train performance: %4.2f%%" % (100.0 * result["train"]))
	print("Test performance:  %4.2f%%" % (100.0 * result["test"]))

def main():
	simple_gnn_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
