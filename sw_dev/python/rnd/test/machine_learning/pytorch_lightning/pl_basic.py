#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch, torchvision
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):
	def __init__(self):
		super().__init__()

		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(28 * 28, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 3)
		)
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(3, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 28 * 28)
		)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def forward(self, x):
		# In lightning, forward defines the prediction/inference actions.
		embedding = self.encoder(x)
		return embedding

	def training_step(self, batch, batch_idx):
		# training_step defines the train loop. It is independent of forward.
		x, y = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)

		loss = torch.nn.functional.mse_loss(x_hat, x)
		#acc = (x_hat == x).float().mean()  # For regression.
		#acc = (y_hat.argmax(dim=-1) == y).float().mean()  # For classification.
		# Logs the accuracy per epoch to TensorBoard (weighted average over batches).
		#self.log("train_acc", acc, on_step=False, on_epoch=True)
		self.log("train_loss", loss)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)

		loss = torch.nn.functional.mse_loss(x_hat, x)
		acc = (x_hat == x).float().mean()  # For regression.
		#acc = (y_hat.argmax(dim=-1) == y).float().mean()  # For classification.
		# By default logs it per epoch (weighted average over batches).
		#self.log("val_acc", acc)
		#self.log("val_loss", loss)
		self.log_dict({"val_loss": loss, "val_acc": acc})

	"""
	def test_step(self, batch, batch_idx):
		x, y = batch
		x = x.view(x.size(0), -1)
		z = self.encoder(x)
		x_hat = self.decoder(z)

		acc = (x_hat == x).float().mean()  # For regression.
		#acc = (y_hat.argmax(dim=-1) == y).float().mean()  # For classification.
		# By default logs it per epoch (weighted average over batches), and returns it afterwards.
		self.log("test_acc", acc)
	"""

# REF [site] >> https://github.com/PyTorchLightning/pytorch-lightning
def simple_autoencoder_example():
	# Data.
	dataset = torchvision.datasets.MNIST("", train=True, download=True, transform=torchvision.transforms.ToTensor())
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
	#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12, persistent_workers=True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
	#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12, persistent_workers=True)

	#--------------------
	# Model.
	model = LitAutoEncoder()

	#--------------------
	# Training.
	#trainer = pl.Trainer()
	trainer = pl.Trainer(gpus=2, num_nodes=1, precision=16, limit_train_batches=0.5)
	#trainer = pl.Trainer(gpus=2, num_nodes=1, precision=16, limit_train_batches=0.5, accelerator="ddp", max_epochs=10)

	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

class LitClassifier(pl.LightningModule):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = torch.nn.functional.cross_entropy(y_hat, y)
		# Logs metrics for each training_step, and the average across the epoch, to the progress bar and logger.
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return {"loss": loss, "pred": y_hat}

	# When training using an accelerator that splits data from each batch across GPUs, sometimes you might need to aggregate them on the main GPU for processing (dp, or ddp2).
	def training_step_end(self, batch_parts):
		# Predictions from each GPU.
		predictions = batch_parts["pred"]
		# Losses from each GPU.
		losses = batch_parts["loss"]

		gpu_0_prediction = predictions[0]
		gpu_1_prediction = predictions[1]

		# Do something with both outputs.
		return (losses[0] + losses[1]) / 2

	# If you need to do something with all the outputs of each training_step, override training_epoch_end yourself.
	def training_epoch_end(self, training_step_outputs):
		for pred in training_step_outputs:
			pass

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = torch.nn.functional.cross_entropy(y_hat, y)
		self.log("val_loss", loss)
		return y_hat

	# When training using an accelerator that splits data from each batch across GPUs, sometimes you might need to aggregate them on the main GPU for processing (dp, or ddp2).
	def validation_step_end(self, batch_parts):
		# Predictions from each GPU.
		predictions = batch_parts["pred"]
		# Losses from each GPU.
		losses = batch_parts["loss"]

		gpu_0_prediction = predictions[0]
		gpu_1_prediction = predictions[1]

		# Do something with both outputs.
		return (losses[0] + losses[1]) / 2

	# If you need to do something with all the outputs of each validation_step, override validation_epoch_end.
	def validation_epoch_end(self, validation_step_outputs):
		for pred in validation_step_outputs:
			pass

	"""
	def test_step(self, batch, batch_idx):

	# When training using an accelerator that splits data from each batch across GPUs, sometimes you might need to aggregate them on the main GPU for processing (dp, or ddp2).
	def test_step_end(self, batch_parts):

	# If you need to do something with all the outputs of each test_step, override test_epoch_end.
	def test_epoch_end(self, validation_step_outputs):
	"""

class Autoencoder(pl.LightningModule):
	def __init__(self, latent_dim=2):
		super().__init__()
		self.save_hyperparameters()
		self.encoder = torch.nn.Sequential(torch.nn.Linear(28 * 28, 256), torch.nn.ReLU(), torch.nn.Linear(256, latent_dim))
		self.decoder = torch.nn.Sequential(torch.nn.Linear(latent_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, 28 * 28))

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.0002)

	def forward(self, x):
		return self.decoder(x)

	def training_step(self, batch, batch_idx):
		loss = self._shared_step(batch)
		# Logs metrics for each training_step, and the average across the epoch, to the progress bar and logger.
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss = self._shared_step(batch)
		self.log("val_loss", loss)

	def test_step(self, batch, batch_idx):
		loss = self._shared_step(batch)
		self.log("test_loss", loss)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		x, _ = batch

		# Encode.
		# For predictions, we could return the embedding or the reconstruction or both based on our need.
		x = x.view(x.size(0), -1)
		return self.encoder(x)
	"""
	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		return self(batch)  # Calls forward().
	"""

	def _shared_step(self, batch):
		x, _ = batch

		# Encode.
		x = x.view(x.size(0), -1)
		z = self.encoder(x)

		# Decode.
		recons = self.decoder(z)

		# Loss.
		recons_loss = torch.nn.functional.mse_loss(recons, x)

		return recons_loss

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
class MNISTDataModule(pl.LightningDataModule):
	import typing

	def __init__(self, data_dir: str = "./"):
		super().__init__()
		self.data_dir = data_dir
		self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

		# Setting default dims here because we know them.
		# Could optionally be assigned dynamically in dm.setup().
		self.dims = (1, 28, 28)

	def prepare_data(self):
		# Download.
		torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
		torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

	def setup(self, stage: typing.Optional[str] = None):
		# Assign train/val split(s) for use in dataloaders.
		if stage in (None, "fit"):
			mnist_full = torchvision.datasets.MNIST(self.data_dir, train=True, transform=self.transform)
			self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [55000, 5000])

			# Optionally...
			#self.dims = tuple(self.mnist_train[0][0].shape)

		# Assign test split(s) for use in dataloader(s).
		if stage in (None, "test"):
			self.mnist_test = torchvision.datasets.MNIST(self.data_dir, train=False, transform=self.transform)

			# Optionally...
			#self.dims = tuple(self.mnist_test[0][0].shape)

	def train_dataloader(self):
		return torch.utils.data.DataLoader(self.mnist_train, batch_size=32)

	def val_dataloader(self):
		return torch.utils.data.DataLoader(self.mnist_val, batch_size=32)

	def test_dataloader(self):
		return torch.utils.data.DataLoader(self.mnist_test, batch_size=32)

	# This is the dataloader that the Trainer predict() method uses.
	def predict_dataloader(self):
		return torch.utils.data.DataLoader(self.mnist_test, batch_size=32)

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
def minimal_example():
	# Data.
	dataset = torchvision.datasets.MNIST("", train=True, download=True, transform=torchvision.transforms.ToTensor())
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])
	test_dataset = torchvision.datasets.MNIST("", train=False, download=True, transform=torchvision.transforms.ToTensor())

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
	#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=12, persistent_workers=True)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
	#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, num_workers=12, persistent_workers=True)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
	#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=12, persistent_workers=True)

	"""
	datamodule = MNISTDataModule()
	#datamodule.prepare_data()
	#datamodule.setup(stage="fit")
	#datamodule.setup(stage="test")
	"""

	#--------------------
	# Model.
	if True:
		#model = LitClassifier(resnet50())
		model = Autoencoder()
	else:
		# Pretrained models.
		model = Autoencoder.load_from_checkpoint("/path/to/checkpoint.ckpt")
		#model = Autoencoder.load_from_checkpoint("/path/to/checkpoint.ckpt", map_location={"cuda:1": "cuda:0"})
		#model = Autoencoder.load_from_checkpoint("/path/to/checkpoint.ckpt", latent_dim=4)  # Hyper-parameters.
		#model = Autoencoder.load_from_checkpoint("/path/to/checkpoint.ckpt", hparams_file="/path/to/hparams_file.yaml")  # Hyper-parameters.
	print("A model created.")

	#--------------------
	# Training.
	print("Training...")
	if True:
		trainer = pl.Trainer(gpus=2, max_epochs=20)
		# Strategy = {"ddp", "ddp_spawn", "deepspeed"}.
		#	REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", auto_select_gpus=False, max_epochs=20, gradient_clip_val=5, gradient_clip_algorithm="norm")
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="ddp", auto_select_gpus=True, max_epochs=20, gradient_clip_val=5, gradient_clip_algorithm="value")
		#trainer = pl.Trainer(gpus=1, precision=16)  # FP16 mixed precision.
		#trainer = pl.Trainer(gpus=1, precision="bf16")  # BFloat16 mixed precision.
		#trainer = pl.Trainer(gpus=1, amp_backend="apex", amp_level="O2")  # NVIDIA APEX mixed precision.
		#trainer = pl.Trainer(auto_lr_find=False, auto_scale_batch_size=False)
		#trainer = pl.Trainer(min_epochs=None, max_epochs=None, min_steps=None, min_steps=-1, max_time=None)
		#trainer = pl.Trainer(logger=True, log_every_n_steps=50, profiler=None)
		#tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs")
		#tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs", name="lightning_logs", version=None, log_graph=False, default_hp_metric=True, prefix="")
		#comet_logger = pl.loggers.CometLogger(save_dir="./comet_logs")
		#trainer = pl.Trainer(logger=True)
		#trainer = pl.Trainer(logger=tensorboard_logger)
		#trainer = pl.Trainer(logger=[tensorboard_logger, comet_logger])
		#trainer = pl.Trainer(default_root_dir="/path/to/checkpoints")  # Saves checkpoints to "/path/to/checkpoints" at every epoch end.
		#trainer = pl.Trainer(resume_from_checkpoint="/path/to/checkpoint.ckpt")  # Resume training. Deprecated.

		# Tunes hyperparameters before training.
		#trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, datamodule=None, scale_batch_size_kwargs=None, lr_find_kwargs=None)

		# Calls pl.LightningModule.training_step() and pl.LightningModule.validation_step().
		trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
		#trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)  # Path/URL of the checkpoint from which training is resumed.
		#trainer.fit(model, datamodule=datamodule)

		#trainer.save_checkpoint("/path/to/checkpoint.ckpt", weights_only=False)
	else:
		# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html

		# A Lightning checkpoint has everything needed to restore a training session including:
		#	16-bit scaling factor (apex).
		#	Current epoch.
		#	Global step.
		#	Model state_dict.
		#	State of all optimizers.
		#	State of all learningRate schedulers.
		#	State of all callbacks.
		#	The hyperparameters used for that model if passed in as hparams (Argparse.Namespace).

		# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing.html
		checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
		"""
		checkpoint_callback = pl.callbacks.ModelCheckpoint(
			monitor="val_loss",
			dirpath="checkpoints",
			filename="model-{epoch:03d}-{val_loss:.2f}",
			save_top_k=3,
			mode="min",
		)
		"""
		#device_stats_callback = DeviceStatsMonitor()
		#early_stopping_callback = pl.callbacks.EarlyStopping("val_loss", min_delta=0.0, patience=3, verbose=False, mode="min", strict=True, check_finite=True, stopping_threshold=None, divergence_threshold=None, check_on_train_epoch_end=None)
		#lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
		#model_pruing_callback = pl.callbacks.ModelPruning(pruning_fn="l1_unstructured", parameters_to_prune=[(model.mlp_1, "weight"), (model.mlp_2, "weight")], parameter_names=None, use_global_unstructured=True, amount=0.5, apply_pruning=True, make_pruning_permanent=True, use_lottery_ticket_hypothesis=True, resample_parameters=False, pruning_dim=None, pruning_norm=None, verbose=0, prune_on_train_epoch_end=True)
		#model_summary_callback = pl.callbacks.ModelSummary(max_depth=1)
		#quantization_aware_training_callback = pl.callbacks.QuantizationAwareTraining(qconfig="fbgemm", observer_type="average", collect_quantization=None, modules_to_fuse=None, input_compatible=True, quantize_on_fit_end=True, observer_enabled_stages=("train",))
		#rich_model_summary_callback = pl.callbacks.RichModelSummary(max_depth=1)
		#rich_progress_bar_callback = pl.callbacks.RichProgressBar(refresh_rate=1, leave=False, theme=RichProgressBarTheme(description="white", progress_bar="#6206E0", progress_bar_finished="#6206E0", progress_bar_pulse="#6206E0", batch_progress="white", time="grey54", processing_speed="grey70", metrics="white"), console_kwargs=None)
		# Stochastic weight averaging (SWA).
		#swa_callback = pl.callbacks.StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, annealing_strategy="cos", avg_fn=None, device=torch.device)
		#timer_callback = pl.callbacks.Timer(duration=None, interval=Interval.step, verbose=True)
		#tqdm_progress_bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1, process_position=0)

		callbacks = [checkpoint_callback]
		#callbacks = [checkpoint_callback, swa_callback]

		trainer = pl.Trainer(gpus=2, max_epochs=20, callbacks=callbacks, enable_checkpointing=True)
		#trainer = pl.Trainer(gpus=2, max_epochs=20, callbacks=None)

		trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
		#trainer.fit(model, datamodule=datamodule)

		#best_model_path = trainer.checkpoint_callback.best_model_path
		#best_model_path = checkpoint_callback.best_model_path

		# All init args were saved to the checkpoint.
		checkpoint = torch.load("/path/to/checkpoint.ckpt")
		print("Hyper-parameters: {}.".format(checkpoint["hyper_parameters"]))
	print("The model trained.")

	#--------------------
	#trainer.validate(model, dataloaders=dataloader, ckpt_path=None, verbose=True)  # Calls pl.LightningModule.validation_step().
	#trainer.test(model, dataloaders=dataloader, ckpt_path=None, verbose=True)  # Calls pl.LightningModule.test_step().
	#trainer.predict(model, dataloaders=dataloader, ckpt_path=None, return_predictions=None)  # Calls pl.LightningModule.predict_step().
	#predictions = model(...)  # Calls pl.LightningModule.forward().

	# Testing.
	print("Testing...")
	if True:
		# Automatically loads the best weights.
		trainer.test(model, dataloaders=test_dataloader)
		#trainer.test(model, datamodule=datamodule)
	else:
		# Automatically auto-loads the best weights.
		trainer.test(dataloaders=test_dataloader)
		#trainer.test(datamodule=datamodule)
	print("The model tested.")

	# Inference.
	print("Inferring...")
	if True:
		trainer.predict(model, dataloaders=test_dataloader)
		#trainer.predict(model, datamodule=datamodule)
	else:
		# When using forward, you are responsible to call eval() and use the no_grad() context manager.
		model.eval()
		model.freeze()
		with torch.no_grad():
			embedding = ...
			reconstruction = model(embedding)
	print("Inferred by the model.")

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/03-initialization-and-optimization.html
def initalization_and_optimization_tutorial():
	import os, copy, math, json
	import urllib.request
	from urllib.error import HTTPError
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.utils.data as data
	from torchvision import transforms
	from torchvision.datasets import FashionMNIST
	import pytorch_lightning as pl
	import seaborn as sns
	#%matplotlib inline
	from matplotlib import cm
	import matplotlib.pyplot as plt
	from IPython.display import set_matplotlib_formats
	from tqdm.notebook import tqdm

	set_matplotlib_formats("svg", "pdf")  # For export.
	sns.set()

	# Path to the folder where the datasets are/should be downloaded (e.g. MNIST).
	DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
	# Path to the folder where the pretrained models are saved.
	CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/InitOptim/")

	# Seed everything.
	pl.seed_everything(42)

	# Ensure that all operations are deterministic on GPU (if used) for reproducibility.
	torch.backends.cudnn.determinstic = True
	torch.backends.cudnn.benchmark = False

	# Fetching the device that will be used throughout this notebook.
	device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
	print("Using device", device)

	# Github URL where saved models are stored for this tutorial.
	base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial4/"
	# Files to download.
	pretrained_files = [
		"FashionMNIST_SGD.config",
		"FashionMNIST_SGD_results.json",
		"FashionMNIST_SGD.tar",
		"FashionMNIST_SGDMom.config",
		"FashionMNIST_SGDMom_results.json",
		"FashionMNIST_SGDMom.tar",
		"FashionMNIST_Adam.config",
		"FashionMNIST_Adam_results.json",
		"FashionMNIST_Adam.tar",
	]
	# Create checkpoint path if it doesn't exist yet.
	os.makedirs(CHECKPOINT_PATH, exist_ok=True)

	# For each file, check whether it already exists. If not, try downloading it.
	for file_name in pretrained_files:
		file_path = os.path.join(CHECKPOINT_PATH, file_name)
		if not os.path.isfile(file_path):
			file_url = base_url + file_name
			print(f"Downloading {file_url}...")
			try:
				urllib.request.urlretrieve(file_url, file_path)
			except HTTPError as e:
				print(
					"Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
					e,
				)

	#--------------------
	# Preparation.

	# Transformations applied on each image => first make them a tensor, then normalize them with mean 0 and std 1.
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))])

	# Loading the training dataset. We need to split it into a training and validation part.
	train_dataset = FashionMNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
	train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])

	# Loading the test set.
	test_set = FashionMNIST(root=DATASET_PATH, train=False, transform=transform, download=True)

	train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
	val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
	test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)

	print("Mean", (train_dataset.data.float() / 255.0).mean().item())
	print("Std", (train_dataset.data.float() / 255.0).std().item())

	imgs, _ = next(iter(train_loader))
	print(f"Mean: {imgs.mean().item():5.3f}")
	print(f"Standard deviation: {imgs.std().item():5.3f}")
	print(f"Maximum: {imgs.max().item():5.3f}")
	print(f"Minimum: {imgs.min().item():5.3f}")

	class BaseNetwork(nn.Module):
		def __init__(self, act_fn, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128]):
			"""
			Args:
				act_fn: Object of the activation function that should be used as non-linearity in the network.
				input_size: Size of the input images in pixels
				num_classes: Number of classes we want to predict
				hidden_sizes: A list of integers specifying the hidden layer sizes in the NN
			"""
			super().__init__()

			# Create the network based on the specified hidden sizes.
			layers = []
			layer_sizes = [input_size] + hidden_sizes
			for layer_index in range(1, len(layer_sizes)):
				layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
			layers += [nn.Linear(layer_sizes[-1], num_classes)]
			# A module list registers a list of modules as submodules (e.g. for parameters).
			self.layers = nn.ModuleList(layers)

			self.config = {
				"act_fn": act_fn.__class__.__name__,
				"input_size": input_size,
				"num_classes": num_classes,
				"hidden_sizes": hidden_sizes,
			}

		def forward(self, x):
			x = x.view(x.size(0), -1)
			for layer in self.layers:
				x = layer(x)
			return x

	class Identity(nn.Module):
		def forward(self, x):
			return x

	act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "identity": Identity}

	##############################################################

	def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
		columns = len(val_dict)
		fig, ax = plt.subplots(1, columns, figsize=(columns * 3, 2.5))
		fig_index = 0
		for key in sorted(val_dict.keys()):
			key_ax = ax[fig_index % columns]
			sns.histplot(
				val_dict[key],
				ax=key_ax,
				color=color,
				bins=50,
				stat=stat,
				kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
			)  # Only plot kde if there is variance.
			hidden_dim_str = (
				r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape) > 1 else ""
			)
			key_ax.set_title(f"{key} {hidden_dim_str}")
			if xlabel is not None:
				key_ax.set_xlabel(xlabel)
			fig_index += 1
		fig.subplots_adjust(wspace=0.4)
		return fig

	##############################################################

	def visualize_weight_distribution(model, color="C0"):
		weights = {}
		for name, param in model.named_parameters():
			if name.endswith(".bias"):
				continue
			key_name = f"Layer {name.split('.')[1]}"
			weights[key_name] = param.detach().view(-1).cpu().numpy()

		# Plotting
		fig = plot_dists(weights, color=color, xlabel="Weight vals")
		fig.suptitle("Weight distribution", fontsize=14, y=1.05)
		plt.show()
		plt.close()

	##############################################################

	def visualize_gradients(model, color="C0", print_variance=False):
		"""
		Args:
			net: Object of class BaseNetwork
			color: Color in which we want to visualize the histogram (for easier separation of activation functions)
		"""
		model.eval()
		small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
		imgs, labels = next(iter(small_loader))
		imgs, labels = imgs.to(device), labels.to(device)

		# Pass one batch through the network, and calculate the gradients for the weights.
		model.zero_grad()
		preds = model(imgs)
		loss = F.cross_entropy(preds, labels)  # Same as nn.CrossEntropyLoss, but as a function instead of module.
		loss.backward()
		# We limit our visualization to the weight parameters and exclude the bias to reduce the number of plots.
		grads = {
			name: params.grad.view(-1).cpu().clone().numpy()
			for name, params in model.named_parameters()
			if "weight" in name
		}
		model.zero_grad()

		# Plotting.
		fig = plot_dists(grads, color=color, xlabel="Grad magnitude")
		fig.suptitle("Gradient distribution", fontsize=14, y=1.05)
		plt.show()
		plt.close()

		if print_variance:
			for key in sorted(grads.keys()):
				print(f"{key} - Variance: {np.var(grads[key])}")

	##############################################################

	def visualize_activations(model, color="C0", print_variance=False):
		model.eval()
		small_loader = data.DataLoader(train_set, batch_size=1024, shuffle=False)
		imgs, labels = next(iter(small_loader))
		imgs, labels = imgs.to(device), labels.to(device)

		# Pass one batch through the network, and calculate the gradients for the weights.
		feats = imgs.view(imgs.shape[0], -1)
		activations = {}
		with torch.no_grad():
			for layer_index, layer in enumerate(model.layers):
				feats = layer(feats)
				if isinstance(layer, nn.Linear):
					activations[f"Layer {layer_index}"] = feats.view(-1).detach().cpu().numpy()

		# Plotting.
		fig = plot_dists(activations, color=color, stat="density", xlabel="Activation vals")
		fig.suptitle("Activation distribution", fontsize=14, y=1.05)
		plt.show()
		plt.close()

		if print_variance:
			for key in sorted(activations.keys()):
				print(f"{key} - Variance: {np.var(activations[key])}")

	#--------------------
	# Initialization.

	model = BaseNetwork(act_fn=Identity()).to(device)

	# Constant initialization.
	def const_init(model, fill=0.0):
		for name, param in model.named_parameters():
			param.data.fill_(fill)

	const_init(model, fill=0.005)
	visualize_gradients(model)
	visualize_activations(model, print_variance=True)

	# Constant variance.
	def var_init(model, std=0.01):
		for name, param in model.named_parameters():
			param.data.normal_(mean=0.0, std=std)

	var_init(model, std=0.01)
	visualize_activations(model, print_variance=True)

	var_init(model, std=0.1)
	visualize_activations(model, print_variance=True)

	# How to find appropriate initialization values.
	def equal_var_init(model):
		for name, param in model.named_parameters():
			if name.endswith(".bias"):
				param.data.fill_(0)
			else:
				param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))

	equal_var_init(model)
	visualize_weight_distribution(model)
	visualize_activations(model, print_variance=True)

	def xavier_init(model):
		for name, param in model.named_parameters():
			if name.endswith(".bias"):
				param.data.fill_(0)
			else:
				bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
				param.data.uniform_(-bound, bound)

	xavier_init(model)
	visualize_gradients(model, print_variance=True)
	visualize_activations(model, print_variance=True)

	model = BaseNetwork(act_fn=nn.Tanh()).to(device)
	xavier_init(model)
	visualize_gradients(model, print_variance=True)
	visualize_activations(model, print_variance=True)

	def kaiming_init(model):
		for name, param in model.named_parameters():
			if name.endswith(".bias"):
				param.data.fill_(0)
			elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input.
				param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
			else:
				param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))

	model = BaseNetwork(act_fn=nn.ReLU()).to(device)
	kaiming_init(model)
	visualize_gradients(model, print_variance=True)
	visualize_activations(model, print_variance=True)

	#--------------------
	# Optimization.

	def _get_config_file(model_path, model_name):
		return os.path.join(model_path, model_name + ".config")

	def _get_model_file(model_path, model_name):
		return os.path.join(model_path, model_name + ".tar")

	def _get_result_file(model_path, model_name):
		return os.path.join(model_path, model_name + "_results.json")

	def load_model(model_path, model_name, net=None):
		config_file = _get_config_file(model_path, model_name)
		model_file = _get_model_file(model_path, model_name)
		assert os.path.isfile(
			config_file
		), f'Could not find the config file "{config_file}". Are you sure this is the correct path and you have your model config stored here?'
		assert os.path.isfile(
			model_file
		), f'Could not find the model file "{model_file}". Are you sure this is the correct path and you have your model stored here?'
		with open(config_file) as f:
			config_dict = json.load(f)
		if net is None:
			act_fn_name = config_dict["act_fn"].pop("name").lower()
			assert (
				act_fn_name in act_fn_by_name
			), f'Unknown activation function "{act_fn_name}". Please add it to the "act_fn_by_name" dict.'
			act_fn = act_fn_by_name[act_fn_name]()
			net = BaseNetwork(act_fn=act_fn, **config_dict)
		net.load_state_dict(torch.load(model_file))
		return net

	def save_model(model, model_path, model_name):
		config_dict = model.config
		os.makedirs(model_path, exist_ok=True)
		config_file = _get_config_file(model_path, model_name)
		model_file = _get_model_file(model_path, model_name)
		with open(config_file, "w") as f:
			json.dump(config_dict, f)
		torch.save(model.state_dict(), model_file)

	def train_model(net, model_name, optim_func, max_epochs=50, batch_size=256, overwrite=False):
		"""Train a model on the training set of FashionMNIST.

		Args:
			net: Object of BaseNetwork
			model_name: (str) Name of the model, used for creating the checkpoint names
			max_epochs: Number of epochs we want to (maximally) train for
			patience: If the performance on the validation set has not improved for #patience epochs, we stop training early
			batch_size: Size of batches used in training
			overwrite: Determines how to handle the case when there already exists a checkpoint. If True, it will be overwritten. Otherwise, we skip training.
		"""
		file_exists = os.path.isfile(_get_model_file(CHECKPOINT_PATH, model_name))
		if file_exists and not overwrite:
			print(f'Model file of "{model_name}" already exists. Skipping training...')
			with open(_get_result_file(CHECKPOINT_PATH, model_name)) as f:
				results = json.load(f)
		else:
			if file_exists:
				print("Model file exists, but will be overwritten...")

			# Defining optimizer, loss and data loader.
			optimizer = optim_func(net.parameters())
			loss_module = nn.CrossEntropyLoss()
			train_loader_local = data.DataLoader(
				train_set, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
			)

			results = None
			val_scores = []
			train_losses, train_scores = [], []
			best_val_epoch = -1
			for epoch in range(max_epochs):
				train_acc, val_acc, epoch_losses = epoch_iteration(
					net, loss_module, optimizer, train_loader_local, val_loader, epoch
				)
				train_scores.append(train_acc)
				val_scores.append(val_acc)
				train_losses += epoch_losses

				if len(val_scores) == 1 or val_acc > val_scores[best_val_epoch]:
					print("\t   (New best performance, saving model...)")
					save_model(net, CHECKPOINT_PATH, model_name)
					best_val_epoch = epoch

		if results is None:
			load_model(CHECKPOINT_PATH, model_name, net=net)
			test_acc = test_model(net, test_loader)
			results = {
				"test_acc": test_acc,
				"val_scores": val_scores,
				"train_losses": train_losses,
				"train_scores": train_scores,
			}
			with open(_get_result_file(CHECKPOINT_PATH, model_name), "w") as f:
				json.dump(results, f)

		# Plot a curve of the validation accuracy.
		sns.set()
		plt.plot([i for i in range(1, len(results["train_scores"]) + 1)], results["train_scores"], label="Train")
		plt.plot([i for i in range(1, len(results["val_scores"]) + 1)], results["val_scores"], label="Val")
		plt.xlabel("Epochs")
		plt.ylabel("Validation accuracy")
		plt.ylim(min(results["val_scores"]), max(results["train_scores"]) * 1.01)
		plt.title(f"Validation performance of {model_name}")
		plt.legend()
		plt.show()
		plt.close()

		print((f" Test accuracy: {results['test_acc']*100.0:4.2f}% ").center(50, "=") + "\n")
		return results

	def epoch_iteration(net, loss_module, optimizer, train_loader_local, val_loader, epoch):
		############
		# Training #
		############
		net.train()
		true_preds, count = 0.0, 0
		epoch_losses = []
		t = tqdm(train_loader_local, leave=False)
		for imgs, labels in t:
			imgs, labels = imgs.to(device), labels.to(device)
			optimizer.zero_grad()
			preds = net(imgs)
			loss = loss_module(preds, labels)
			loss.backward()
			optimizer.step()
			# Record statistics during training.
			true_preds += (preds.argmax(dim=-1) == labels).sum().item()
			count += labels.shape[0]
			t.set_description(f"Epoch {epoch+1}: loss={loss.item():4.2f}")
			epoch_losses.append(loss.item())
		train_acc = true_preds / count

		##############
		# Validation #
		##############
		val_acc = test_model(net, val_loader)
		print(
			f"[Epoch {epoch+1:2i}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%"
		)
		return train_acc, val_acc, epoch_losses

	def test_model(net, data_loader):
		"""Test a model on a specified dataset.

		Args:
			net: Trained model of type BaseNetwork
			data_loader: DataLoader object of the dataset to test on (validation or test)
		"""
		net.eval()
		true_preds, count = 0.0, 0
		for imgs, labels in data_loader:
			imgs, labels = imgs.to(device), labels.to(device)
			with torch.no_grad():
				preds = net(imgs).argmax(dim=-1)
				true_preds += (preds == labels).sum().item()
				count += labels.shape[0]
		test_acc = true_preds / count
		return test_acc

	class OptimizerTemplate:
		def __init__(self, params, lr):
			self.params = list(params)
			self.lr = lr

		def zero_grad(self):
			# Set gradients of all parameters to zero.
			for p in self.params:
				if p.grad is not None:
					p.grad.detach_()  # For second-order optimizers important.
					p.grad.zero_()

		@torch.no_grad()
		def step(self):
			# Apply update step to all parameters.
			for p in self.params:
				if p.grad is None:  # We skip parameters without any gradients.
					continue
				self.update_param(p)

		def update_param(self, p):
			# To be implemented in optimizer-specific classes.
			raise NotImplementedError

	class SGD(OptimizerTemplate):
		def __init__(self, params, lr):
			super().__init__(params, lr)

		def update_param(self, p):
			p_update = -self.lr * p.grad
			p.add_(p_update)  # In-place update => saves memory and does not create computation graph.

	class SGDMomentum(OptimizerTemplate):
		def __init__(self, params, lr, momentum=0.0):
			super().__init__(params, lr)
			self.momentum = momentum  # Corresponds to beta_1 in the equation above.
			self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}  # Dict to store m_t.

		def update_param(self, p):
			self.param_momentum[p] = (1 - self.momentum) * p.grad + self.momentum * self.param_momentum[p]
			p_update = -self.lr * self.param_momentum[p]
			p.add_(p_update)

	class Adam(OptimizerTemplate):
		def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
			super().__init__(params, lr)
			self.beta1 = beta1
			self.beta2 = beta2
			self.eps = eps
			self.param_step = {p: 0 for p in self.params}  # Remembers "t" for each parameter for bias correction.
			self.param_momentum = {p: torch.zeros_like(p.data) for p in self.params}
			self.param_2nd_momentum = {p: torch.zeros_like(p.data) for p in self.params}

		def update_param(self, p):
			self.param_step[p] += 1

			self.param_momentum[p] = (1 - self.beta1) * p.grad + self.beta1 * self.param_momentum[p]
			self.param_2nd_momentum[p] = (1 - self.beta2) * (p.grad) ** 2 + self.beta2 * self.param_2nd_momentum[p]

			bias_correction_1 = 1 - self.beta1 ** self.param_step[p]
			bias_correction_2 = 1 - self.beta2 ** self.param_step[p]

			p_2nd_mom = self.param_2nd_momentum[p] / bias_correction_2
			p_mom = self.param_momentum[p] / bias_correction_1
			p_lr = self.lr / (torch.sqrt(p_2nd_mom) + self.eps)
			p_update = -p_lr * p_mom

			p.add_(p_update)

	#--------------------
	# Comparing optimizers on model training.

	base_model = BaseNetwork(act_fn=nn.ReLU(), hidden_sizes=[512, 256, 256, 128])
	kaiming_init(base_model)

	SGD_model = copy.deepcopy(base_model).to(device)
	SGD_results = train_model(
		SGD_model, "FashionMNIST_SGD", lambda params: SGD(params, lr=1e-1), max_epochs=40, batch_size=256
	)

	SGDMom_model = copy.deepcopy(base_model).to(device)
	SGDMom_results = train_model(
		SGDMom_model,
		"FashionMNIST_SGDMom",
		lambda params: SGDMomentum(params, lr=1e-1, momentum=0.9),
		max_epochs=40,
		batch_size=256,
	)

	Adam_model = copy.deepcopy(base_model).to(device)
	Adam_results = train_model(
		Adam_model, "FashionMNIST_Adam", lambda params: Adam(params, lr=1e-3), max_epochs=40, batch_size=256
	)

	#--------------------
	# Pathological curvatures.

	def pathological_curve_loss(w1, w2):
		# Example of a pathological curvature. There are many more possible, feel free to experiment here!
		x1_loss = torch.tanh(w1) ** 2 + 0.01 * torch.abs(w1)
		x2_loss = torch.sigmoid(w2)
		return x1_loss + x2_loss

	def plot_curve(
		curve_fn, x_range=(-5, 5), y_range=(-5, 5), plot_3d=False, cmap=cm.viridis, title="Pathological curvature"
	):
		fig = plt.figure()
		ax = fig.gca(projection="3d") if plot_3d else fig.gca()

		x = torch.arange(x_range[0], x_range[1], (x_range[1] - x_range[0]) / 100.0)
		y = torch.arange(y_range[0], y_range[1], (y_range[1] - y_range[0]) / 100.0)
		x, y = torch.meshgrid([x, y])
		z = curve_fn(x, y)
		x, y, z = x.numpy(), y.numpy(), z.numpy()

		if plot_3d:
			ax.plot_surface(x, y, z, cmap=cmap, linewidth=1, color="#000", antialiased=False)
			ax.set_zlabel("loss")
		else:
			ax.imshow(z.T[::-1], cmap=cmap, extent=(x_range[0], x_range[1], y_range[0], y_range[1]))
		plt.title(title)
		ax.set_xlabel(r"$w_1$")
		ax.set_ylabel(r"$w_2$")
		plt.tight_layout()
		return ax

	sns.reset_orig()
	_ = plot_curve(pathological_curve_loss, plot_3d=True)
	plt.show()

	def train_curve(optimizer_func, curve_func=pathological_curve_loss, num_updates=100, init=[5, 5]):
		"""
		Args:
			optimizer_func: Constructor of the optimizer to use. Should only take a parameter list
			curve_func: Loss function (e.g. pathological curvature)
			num_updates: Number of updates/steps to take when optimizing
			init: Initial values of parameters. Must be a list/tuple with two elements representing w_1 and w_2
		Returns:
			Numpy array of shape [num_updates, 3] with [t,:2] being the parameter values at step t, and [t,2] the loss at t.
		"""
		weights = nn.Parameter(torch.FloatTensor(init), requires_grad=True)
		optim = optimizer_func([weights])

		list_points = []
		for _ in range(num_updates):
			loss = curve_func(weights[0], weights[1])
			list_points.append(torch.cat([weights.data.detach(), loss.unsqueeze(dim=0).detach()], dim=0))
			optim.zero_grad()
			loss.backward()
			optim.step()
		points = torch.stack(list_points, dim=0).numpy()
		return points

	SGD_points = train_curve(lambda params: SGD(params, lr=10))
	SGDMom_points = train_curve(lambda params: SGDMomentum(params, lr=10, momentum=0.9))
	Adam_points = train_curve(lambda params: Adam(params, lr=1))

	all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
	ax = plot_curve(
		pathological_curve_loss,
		x_range=(-np.absolute(all_points[:, 0]).max(), np.absolute(all_points[:, 0]).max()),
		y_range=(all_points[:, 1].min(), all_points[:, 1].max()),
		plot_3d=False,
	)
	ax.plot(SGD_points[:, 0], SGD_points[:, 1], color="red", marker="o", zorder=1, label="SGD")
	ax.plot(SGDMom_points[:, 0], SGDMom_points[:, 1], color="blue", marker="o", zorder=2, label="SGDMom")
	ax.plot(Adam_points[:, 0], Adam_points[:, 1], color="grey", marker="o", zorder=3, label="Adam")
	plt.legend()
	plt.show()

	#--------------------
	# Steep optima.

	def bivar_gaussian(w1, w2, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
		norm = 1 / (2 * np.pi * x_sig * y_sig)
		x_exp = (-1 * (w1 - x_mean) ** 2) / (2 * x_sig ** 2)
		y_exp = (-1 * (w2 - y_mean) ** 2) / (2 * y_sig ** 2)
		return norm * torch.exp(x_exp + y_exp)

	def comb_func(w1, w2):
		z = -bivar_gaussian(w1, w2, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
		z -= bivar_gaussian(w1, w2, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
		z -= bivar_gaussian(w1, w2, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
		return z

	_ = plot_curve(comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=True, title="Steep optima")

	SGD_points = train_curve(lambda params: SGD(params, lr=0.5), comb_func, init=[0, 0])
	SGDMom_points = train_curve(lambda params: SGDMomentum(params, lr=1, momentum=0.9), comb_func, init=[0, 0])
	Adam_points = train_curve(lambda params: Adam(params, lr=0.2), comb_func, init=[0, 0])

	all_points = np.concatenate([SGD_points, SGDMom_points, Adam_points], axis=0)
	ax = plot_curve(comb_func, x_range=(-2, 2), y_range=(-2, 2), plot_3d=False, title="Steep optima")
	ax.plot(SGD_points[:, 0], SGD_points[:, 1], color="red", marker="o", zorder=3, label="SGD", alpha=0.7)
	ax.plot(SGDMom_points[:, 0], SGDMom_points[:, 1], color="blue", marker="o", zorder=2, label="SGDMom", alpha=0.7)
	ax.plot(Adam_points[:, 0], Adam_points[:, 1], color="grey", marker="o", zorder=1, label="Adam", alpha=0.7)
	ax.set_xlim(-2, 2)
	ax.set_ylim(-2, 2)
	plt.legend()
	plt.show()

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/04-inception-resnet-densenet.html
def inception_resnet_densenet_tutorial():
	import os
	from types import SimpleNamespace
	import urllib.request
	from urllib.error import HTTPError
	import numpy as np
	import torch
	import torch.nn as nn
	import torch.optim as optim
	import torch.utils.data as data
	import torchvision
	from torchvision import transforms
	from torchvision.datasets import CIFAR10
	import pytorch_lightning as pl
	from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
	import seaborn as sns
	import tabulate
	import matplotlib
	import matplotlib.pyplot as plt
	#%matplotlib inline
	from IPython.display import HTML, display, set_matplotlib_formats
	from PIL import Image

	set_matplotlib_formats("svg", "pdf")  # For export.
	matplotlib.rcParams["lines.linewidth"] = 2.0
	sns.reset_orig()

	#--------------------
	# PyTorch.
	# Torchvision.

	# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10).
	DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
	# Path to the folder where the pretrained models are saved.
	CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ConvNets")

	# Function for setting the seed.
	pl.seed_everything(42)

	# Ensure that all operations are deterministic on GPU (if used) for reproducibility.
	torch.backends.cudnn.determinstic = True
	torch.backends.cudnn.benchmark = False

	device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

	# Github URL where saved models are stored for this tutorial.
	base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial5/"
	# Files to download.
	pretrained_files = [
		"GoogleNet.ckpt",
		"ResNet.ckpt",
		"ResNetPreAct.ckpt",
		"DenseNet.ckpt",
		"tensorboards/GoogleNet/events.out.tfevents.googlenet",
		"tensorboards/ResNet/events.out.tfevents.resnet",
		"tensorboards/ResNetPreAct/events.out.tfevents.resnetpreact",
		"tensorboards/DenseNet/events.out.tfevents.densenet",
	]
	# Create checkpoint path if it doesn't exist yet.
	os.makedirs(CHECKPOINT_PATH, exist_ok=True)

	# For each file, check whether it already exists. If not, try downloading it.
	for file_name in pretrained_files:
		file_path = os.path.join(CHECKPOINT_PATH, file_name)
		if "/" in file_name:
			os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
		if not os.path.isfile(file_path):
			file_url = base_url + file_name
			print(f"Downloading {file_url}...")
			try:
				urllib.request.urlretrieve(file_url, file_path)
			except HTTPError as e:
				print(
					"Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
					e,
				)

	train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
	DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
	DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
	print("Data mean", DATA_MEANS)
	print("Data std", DATA_STD)

	test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)])
	# For training, we add some augmentation. Networks are too powerful and would overfit.
	train_transform = transforms.Compose(
		[
			transforms.RandomHorizontalFlip(),
			transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
			transforms.ToTensor(),
			transforms.Normalize(DATA_MEANS, DATA_STD),
		]
	)
	# Loading the training dataset. We need to split it into a training and validation part.
	# We need to do a little trick because the validation set should not use the augmentation.
	train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
	val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
	pl.seed_everything(42)
	train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
	pl.seed_everything(42)
	_, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

	# Loading the test set.
	test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

	# We define a set of data loaders that we can use for various purposes later.
	train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
	val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
	test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

	imgs, _ = next(iter(train_loader))
	print("Batch mean", imgs.mean(dim=[0, 2, 3]))
	print("Batch std", imgs.std(dim=[0, 2, 3]))

	NUM_IMAGES = 4
	images = [train_dataset[idx][0] for idx in range(NUM_IMAGES)]
	orig_images = [Image.fromarray(train_dataset.data[idx]) for idx in range(NUM_IMAGES)]
	orig_images = [test_transform(img) for img in orig_images]

	img_grid = torchvision.utils.make_grid(torch.stack(images + orig_images, dim=0), nrow=4, normalize=True, pad_value=0.5)
	img_grid = img_grid.permute(1, 2, 0)

	plt.figure(figsize=(8, 8))
	plt.title("Augmentation examples on CIFAR10")
	plt.imshow(img_grid)
	plt.axis("off")
	plt.show()
	plt.close()

	#--------------------
	# PyTorch Lightning.

	# Setting the seed.
	pl.seed_everything(42)

	model_dict = {}

	def create_model(model_name, model_hparams):
		if model_name in model_dict:
			return model_dict[model_name](**model_hparams)
		else:
			assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

	class CIFARModule(pl.LightningModule):
		def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
			"""
			Inputs:
				model_name - Name of the model/CNN to run. Used for creating the model (see function below)
				model_hparams - Hyperparameters for the model, as dictionary.
				optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
				optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
			"""
			super().__init__()
			# Exports the hyperparameters to a YAML file, and create "self.hparams" namespace.
			self.save_hyperparameters()
			# Create model.
			self.model = create_model(model_name, model_hparams)
			# Create loss module.
			self.loss_module = nn.CrossEntropyLoss()
			# Example input for visualizing the graph in Tensorboard.
			self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

		def configure_optimizers(self):
			# We will support Adam or SGD as optimizers.
			if self.hparams.optimizer_name == "Adam":
				# AdamW is Adam with a correct implementation of weight decay (see here
				# for details: https://arxiv.org/pdf/1711.05101.pdf).
				optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
			elif self.hparams.optimizer_name == "SGD":
				optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
			else:
				assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

			# We will reduce the learning rate by 0.1 after 100 and 150 epochs.
			scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
			return [optimizer], [scheduler]
			#return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
			#return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val_f1", "reduce_on_plateau": False, "frequency": 1}]
			#return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

		def forward(self, imgs):
			# Forward function that is run when visualizing the graph.
			return self.model(imgs)

		def training_step(self, batch, batch_idx):
			# "batch" is the output of the training data loader.
			imgs, labels = batch
			preds = self.model(imgs)
			loss = self.loss_module(preds, labels)
			acc = (preds.argmax(dim=-1) == labels).float().mean()

			# Logs the accuracy per epoch to tensorboard (weighted average over batches).
			self.log("train_acc", acc, on_step=False, on_epoch=True)
			self.log("train_loss", loss)
			return loss  # Return tensor to call ".backward" on.

		def validation_step(self, batch, batch_idx):
			imgs, labels = batch
			preds = self.model(imgs).argmax(dim=-1)
			acc = (labels == preds).float().mean()
			# By default logs it per epoch (weighted average over batches).
			self.log("val_acc", acc)

		def test_step(self, batch, batch_idx):
			imgs, labels = batch
			preds = self.model(imgs).argmax(dim=-1)
			acc = (labels == preds).float().mean()
			# By default logs it per epoch (weighted average over batches), and returns it afterwards.
			self.log("test_acc", acc)

	# Callbacks.
	# Another important part of PyTorch Lightning is the concept of callbacks.
	# Callbacks are self-contained functions that contain the non-essential logic of your Lightning Module.
	# They are usually called after finishing a training epoch, but can also influence other parts of your training loop.
	# For instance, we will use the following two pre-defined callbacks: LearningRateMonitor and ModelCheckpoint.
	# The learning rate monitor adds the current learning rate to our TensorBoard, which helps to verify that our learning rate scheduler works correctly.
	# The model checkpoint callback allows you to customize the saving routine of your checkpoints.
	# For instance, how many checkpoints to keep, when to save, which metric to look out for, etc.

	act_fn_by_name = {"tanh": nn.Tanh, "relu": nn.ReLU, "leakyrelu": nn.LeakyReLU, "gelu": nn.GELU}

	def train_model(model_name, save_name=None, **kwargs):
		"""
		Inputs:
			model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
			save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
		"""
		if save_name is None:
			save_name = model_name

		# Create a PyTorch Lightning trainer with the generation callback.
		trainer = pl.Trainer(
			default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),  # Where to save models.
			# We run on a single GPU (if possible).
			gpus=1 if str(device) == "cuda:0" else 0,
			# How many epochs to train for if no patience is set.
			max_epochs=180,
			callbacks=[
				ModelCheckpoint(
					save_weights_only=True, mode="max", monitor="val_acc"
				),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer.
				LearningRateMonitor("epoch"),
			],  # Log learning rate every epoch.
			progress_bar_refresh_rate=1,
		)  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate.
		trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard.
		trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need.

		# Check whether pretrained model exists. If yes, load it and skip training.
		pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
		if os.path.isfile(pretrained_filename):
			print(f"Found pretrained model at {pretrained_filename}, loading...")
			# Automatically loads the model with the saved hyperparameters.
			model = CIFARModule.load_from_checkpoint(pretrained_filename)
		else:
			pl.seed_everything(42)  # To be reproducable.
			model = CIFARModule(model_name=model_name, **kwargs)
			trainer.fit(model, train_loader, val_loader)
			model = CIFARModule.load_from_checkpoint(
				trainer.checkpoint_callback.best_model_path
			)  # Load best checkpoint after training.

		# Test best model on validation and test set.
		val_result = trainer.validate(model, dataloaders=val_loader, verbose=False)
		test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
		result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

		return model, result

	#--------------------
	# Inception.

	class InceptionBlock(nn.Module):
		def __init__(self, c_in, c_red: dict, c_out: dict, act_fn):
			"""
			Inputs:
				c_in - Number of input feature maps from the previous layers
				c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
				c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
				act_fn - Activation class constructor (e.g. nn.ReLU)
			"""
			super().__init__()

			# 1x1 convolution branch.
			self.conv_1x1 = nn.Sequential(
				nn.Conv2d(c_in, c_out["1x1"], kernel_size=1), nn.BatchNorm2d(c_out["1x1"]), act_fn()
			)

			# 3x3 convolution branch.
			self.conv_3x3 = nn.Sequential(
				nn.Conv2d(c_in, c_red["3x3"], kernel_size=1),
				nn.BatchNorm2d(c_red["3x3"]),
				act_fn(),
				nn.Conv2d(c_red["3x3"], c_out["3x3"], kernel_size=3, padding=1),
				nn.BatchNorm2d(c_out["3x3"]),
				act_fn(),
			)

			# 5x5 convolution branch.
			self.conv_5x5 = nn.Sequential(
				nn.Conv2d(c_in, c_red["5x5"], kernel_size=1),
				nn.BatchNorm2d(c_red["5x5"]),
				act_fn(),
				nn.Conv2d(c_red["5x5"], c_out["5x5"], kernel_size=5, padding=2),
				nn.BatchNorm2d(c_out["5x5"]),
				act_fn(),
			)

			# Max-pool branch.
			self.max_pool = nn.Sequential(
				nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
				nn.Conv2d(c_in, c_out["max"], kernel_size=1),
				nn.BatchNorm2d(c_out["max"]),
				act_fn(),
			)

		def forward(self, x):
			x_1x1 = self.conv_1x1(x)
			x_3x3 = self.conv_3x3(x)
			x_5x5 = self.conv_5x5(x)
			x_max = self.max_pool(x)
			x_out = torch.cat([x_1x1, x_3x3, x_5x5, x_max], dim=1)
			return x_out

	class GoogleNet(nn.Module):
		def __init__(self, num_classes=10, act_fn_name="relu", **kwargs):
			super().__init__()
			self.hparams = SimpleNamespace(
				num_classes=num_classes, act_fn_name=act_fn_name, act_fn=act_fn_by_name[act_fn_name]
			)
			self._create_network()
			self._init_params()

		def _create_network(self):
			# A first convolution on the original image to scale up the channel size.
			self.input_net = nn.Sequential(
				nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), self.hparams.act_fn()
			)
			# Stacking inception blocks.
			self.inception_blocks = nn.Sequential(
				InceptionBlock(
					64,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 16, "3x3": 32, "5x5": 8, "max": 8},
					act_fn=self.hparams.act_fn,
				),
				InceptionBlock(
					64,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
					act_fn=self.hparams.act_fn,
				),
				nn.MaxPool2d(3, stride=2, padding=1),  # 32x32 => 16x16.
				InceptionBlock(
					96,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 24, "3x3": 48, "5x5": 12, "max": 12},
					act_fn=self.hparams.act_fn,
				),
				InceptionBlock(
					96,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
					act_fn=self.hparams.act_fn,
				),
				InceptionBlock(
					96,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 16, "3x3": 48, "5x5": 16, "max": 16},
					act_fn=self.hparams.act_fn,
				),
				InceptionBlock(
					96,
					c_red={"3x3": 32, "5x5": 16},
					c_out={"1x1": 32, "3x3": 48, "5x5": 24, "max": 24},
					act_fn=self.hparams.act_fn,
				),
				nn.MaxPool2d(3, stride=2, padding=1),  # 16x16 => 8x8.
				InceptionBlock(
					128,
					c_red={"3x3": 48, "5x5": 16},
					c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
					act_fn=self.hparams.act_fn,
				),
				InceptionBlock(
					128,
					c_red={"3x3": 48, "5x5": 16},
					c_out={"1x1": 32, "3x3": 64, "5x5": 16, "max": 16},
					act_fn=self.hparams.act_fn,
				),
			)
			# Mapping to classification output.
			self.output_net = nn.Sequential(
				nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, self.hparams.num_classes)
			)

		def _init_params(self):
			# Based on our discussion in Tutorial 4, we should initialize the
			# convolutions according to the activation function.
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

		def forward(self, x):
			x = self.input_net(x)
			x = self.inception_blocks(x)
			x = self.output_net(x)
			return x

	model_dict["GoogleNet"] = GoogleNet

	googlenet_model, googlenet_results = train_model(
		model_name="GoogleNet",
		model_hparams={"num_classes": 10, "act_fn_name": "relu"},
		optimizer_name="Adam",
		optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
	)

	print("GoogleNet Results", googlenet_results)

	# Tensorboard log.

	# Import tensorboard.
	# %load_ext tensorboard

	# Opens tensorboard in notebook. Adjust the path to your CHECKPOINT_PATH!
	# %tensorboard --logdir ../saved_models/tutorial5/tensorboards/GoogleNet/

	#--------------------
	# ResNet.

	class ResNetBlock(nn.Module):
		def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
			"""
			Inputs:
				c_in - Number of input features
				act_fn - Activation class constructor (e.g. nn.ReLU)
				subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
				c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
			"""
			super().__init__()
			if not subsample:
				c_out = c_in

			# Network representing F.
			self.net = nn.Sequential(
				nn.Conv2d(
					c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False
				),  # No bias needed as the Batch Norm handles it.
				nn.BatchNorm2d(c_out),
				act_fn(),
				nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(c_out),
			)

			# 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size.
			self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
			self.act_fn = act_fn()

		def forward(self, x):
			z = self.net(x)
			if self.downsample is not None:
				x = self.downsample(x)
			out = z + x
			out = self.act_fn(out)
			return out

	class PreActResNetBlock(nn.Module):
		def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
			"""
			Inputs:
				c_in - Number of input features
				act_fn - Activation class constructor (e.g. nn.ReLU)
				subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
				c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
			"""
			super().__init__()
			if not subsample:
				c_out = c_in

			# Network representing F.
			self.net = nn.Sequential(
				nn.BatchNorm2d(c_in),
				act_fn(),
				nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
				nn.BatchNorm2d(c_out),
				act_fn(),
				nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
			)

			# 1x1 convolution needs to apply non-linearity as well as not done on skip connection.
			self.downsample = (
				nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
				if subsample
				else None
			)

		def forward(self, x):
			z = self.net(x)
			if self.downsample is not None:
				x = self.downsample(x)
			out = z + x
			return out

	resnet_blocks_by_name = {"ResNetBlock": ResNetBlock, "PreActResNetBlock": PreActResNetBlock}

	class ResNet(nn.Module):
		def __init__(
			self,
			num_classes=10,
			num_blocks=[3, 3, 3],
			c_hidden=[16, 32, 64],
			act_fn_name="relu",
			block_name="ResNetBlock",
			**kwargs,
		):
			"""
			Inputs:
				num_classes - Number of classification outputs (10 for CIFAR10)
				num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling, except the first.
				c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
				act_fn_name - Name of the activation function to use, looked up in "act_fn_by_name"
				block_name - Name of the ResNet block, looked up in "resnet_blocks_by_name"
			"""
			super().__init__()
			assert block_name in resnet_blocks_by_name
			self.hparams = SimpleNamespace(
				num_classes=num_classes,
				c_hidden=c_hidden,
				num_blocks=num_blocks,
				act_fn_name=act_fn_name,
				act_fn=act_fn_by_name[act_fn_name],
				block_class=resnet_blocks_by_name[block_name],
			)
			self._create_network()
			self._init_params()

		def _create_network(self):
			c_hidden = self.hparams.c_hidden

			# A first convolution on the original image to scale up the channel size.
			if self.hparams.block_class == PreActResNetBlock:  # => Don't apply non-linearity on output.
				self.input_net = nn.Sequential(nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False))
			else:
				self.input_net = nn.Sequential(
					nn.Conv2d(3, c_hidden[0], kernel_size=3, padding=1, bias=False),
					nn.BatchNorm2d(c_hidden[0]),
					self.hparams.act_fn(),
				)

			# Creating the ResNet blocks.
			blocks = []
			for block_idx, block_count in enumerate(self.hparams.num_blocks):
				for bc in range(block_count):
					# Subsample the first block of each group, except the very first one.
					subsample = bc == 0 and block_idx > 0
					blocks.append(
						self.hparams.block_class(
							c_in=c_hidden[block_idx if not subsample else (block_idx - 1)],
							act_fn=self.hparams.act_fn,
							subsample=subsample,
							c_out=c_hidden[block_idx],
						)
					)
			self.blocks = nn.Sequential(*blocks)

			# Mapping to classification output.
			self.output_net = nn.Sequential(
				nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(c_hidden[-1], self.hparams.num_classes)
			)

		def _init_params(self):
			# Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function.
			# Fan-out focuses on the gradient distribution, and is commonly used in ResNets.
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.hparams.act_fn_name)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

		def forward(self, x):
			x = self.input_net(x)
			x = self.blocks(x)
			x = self.output_net(x)
			return x

	model_dict["ResNet"] = ResNet

	resnet_model, resnet_results = train_model(
		model_name="ResNet",
		model_hparams={"num_classes": 10, "c_hidden": [16, 32, 64], "num_blocks": [3, 3, 3], "act_fn_name": "relu"},
		optimizer_name="SGD",
		optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
	)

	resnetpreact_model, resnetpreact_results = train_model(
		model_name="ResNet",
		model_hparams={
			"num_classes": 10,
			"c_hidden": [16, 32, 64],
			"num_blocks": [3, 3, 3],
			"act_fn_name": "relu",
			"block_name": "PreActResNetBlock",
		},
		optimizer_name="SGD",
		optimizer_hparams={"lr": 0.1, "momentum": 0.9, "weight_decay": 1e-4},
		save_name="ResNetPreAct",
	)

	# Tensorboard log.

	# Opens tensorboard in notebook. Adjust the path to your CHECKPOINT_PATH! Feel free to change "ResNet" to "ResNetPreAct".
	# %tensorboard --logdir ../saved_models/tutorial5/tensorboards/ResNet/

	#--------------------
	# DenseNet.

	class DenseLayer(nn.Module):
		def __init__(self, c_in, bn_size, growth_rate, act_fn):
			"""
			Inputs:
				c_in - Number of input channels
				bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
				growth_rate - Number of output channels of the 3x3 convolution
				act_fn - Activation class constructor (e.g. nn.ReLU)
			"""
			super().__init__()
			self.net = nn.Sequential(
				nn.BatchNorm2d(c_in),
				act_fn(),
				nn.Conv2d(c_in, bn_size * growth_rate, kernel_size=1, bias=False),
				nn.BatchNorm2d(bn_size * growth_rate),
				act_fn(),
				nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
			)

		def forward(self, x):
			out = self.net(x)
			out = torch.cat([out, x], dim=1)
			return out

	class DenseBlock(nn.Module):
		def __init__(self, c_in, num_layers, bn_size, growth_rate, act_fn):
			"""
			Inputs:
				c_in - Number of input channels
				num_layers - Number of dense layers to apply in the block
				bn_size - Bottleneck size to use in the dense layers
				growth_rate - Growth rate to use in the dense layers
				act_fn - Activation function to use in the dense layers
			"""
			super().__init__()
			layers = []
			for layer_idx in range(num_layers):
				# Input channels are original plus the feature maps from previous layers.
				layer_c_in = c_in + layer_idx * growth_rate
				layers.append(DenseLayer(c_in=layer_c_in, bn_size=bn_size, growth_rate=growth_rate, act_fn=act_fn))
			self.block = nn.Sequential(*layers)

		def forward(self, x):
			out = self.block(x)
			return out

	class TransitionLayer(nn.Module):
		def __init__(self, c_in, c_out, act_fn):
			super().__init__()
			self.transition = nn.Sequential(
				nn.BatchNorm2d(c_in),
				act_fn(),
				nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
				nn.AvgPool2d(kernel_size=2, stride=2),  # Average the output for each 2x2 pixel group.
			)

		def forward(self, x):
			return self.transition(x)

	class DenseNet(nn.Module):
		def __init__(
			self, num_classes=10, num_layers=[6, 6, 6, 6], bn_size=2, growth_rate=16, act_fn_name="relu", **kwargs
		):
			super().__init__()
			self.hparams = SimpleNamespace(
				num_classes=num_classes,
				num_layers=num_layers,
				bn_size=bn_size,
				growth_rate=growth_rate,
				act_fn_name=act_fn_name,
				act_fn=act_fn_by_name[act_fn_name],
			)
			self._create_network()
			self._init_params()

		def _create_network(self):
			c_hidden = self.hparams.growth_rate * self.hparams.bn_size  # The start number of hidden channels.

			# A first convolution on the original image to scale up the channel size.
			self.input_net = nn.Sequential(
				# No batch norm or activation function as done inside the Dense layers.
				nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
			)

			# Creating the dense blocks, eventually including transition layers.
			blocks = []
			for block_idx, num_layers in enumerate(self.hparams.num_layers):
				blocks.append(
					DenseBlock(
						c_in=c_hidden,
						num_layers=num_layers,
						bn_size=self.hparams.bn_size,
						growth_rate=self.hparams.growth_rate,
						act_fn=self.hparams.act_fn,
					)
				)
				c_hidden = c_hidden + num_layers * self.hparams.growth_rate  # Overall output of the dense block.
				if block_idx < len(self.hparams.num_layers) - 1:  # Don't apply transition layer on last block.
					blocks.append(TransitionLayer(c_in=c_hidden, c_out=c_hidden // 2, act_fn=self.hparams.act_fn))
					c_hidden = c_hidden // 2

			self.blocks = nn.Sequential(*blocks)

			# Mapping to classification output.
			self.output_net = nn.Sequential(
				nn.BatchNorm2d(c_hidden),  # The features have not passed a non-linearity until here.
				self.hparams.act_fn(),
				nn.AdaptiveAvgPool2d((1, 1)),
				nn.Flatten(),
				nn.Linear(c_hidden, self.hparams.num_classes),
			)

		def _init_params(self):
			# Based on our discussion in Tutorial 4, we should initialize the
			# convolutions according to the activation function.
			for m in self.modules():
				if isinstance(m, nn.Conv2d):
					nn.init.kaiming_normal_(m.weight, nonlinearity=self.hparams.act_fn_name)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

		def forward(self, x):
			x = self.input_net(x)
			x = self.blocks(x)
			x = self.output_net(x)
			return x

	model_dict["DenseNet"] = DenseNet

	densenet_model, densenet_results = train_model(
		model_name="DenseNet",
		model_hparams={
			"num_classes": 10,
			"num_layers": [6, 6, 6, 6],
			"bn_size": 2,
			"growth_rate": 16,
			"act_fn_name": "relu",
		},
		optimizer_name="Adam",
		optimizer_hparams={"lr": 1e-3, "weight_decay": 1e-4},
	)

	# Tensorboard log.

	# Opens tensorboard in notebook. Adjust the path to your CHECKPOINT_PATH! Feel free to change "ResNet" to "ResNetPreAct".
	# %tensorboard --logdir ../saved_models/tutorial5/tensorboards/DenseNet/

	#--------------------
	# Conclusion and comparison.

	all_models = [
		("GoogleNet", googlenet_results, googlenet_model),
		("ResNet", resnet_results, resnet_model),
		("ResNetPreAct", resnetpreact_results, resnetpreact_model),
		("DenseNet", densenet_results, densenet_model),
	]
	table = [
		[
			model_name,
			f"{100.0*model_results['val']:4.2f}%",
			f"{100.0*model_results['test']:4.2f}%",
			f"{sum(np.prod(p.shape) for p in model.parameters()):,}",
		]
		for model_name, model_results, model in all_models
	]
	display(
		HTML(
			tabulate.tabulate(table, tablefmt="html", headers=["Model", "Val Accuracy", "Test Accuracy", "Num Parameters"])
		)
	)

def main():
	#simple_autoencoder_example()
	minimal_example()

	#initalization_and_optimization_tutorial()
	#inception_resnet_densenet_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
