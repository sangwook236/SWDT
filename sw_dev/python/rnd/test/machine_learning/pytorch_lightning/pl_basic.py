#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import typing, math, random, re, unicodedata, difflib, time
#from tkinter import Y
import torch, torchvision
import pytorch_lightning as pl

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
		params = [p for p in self.parameters() if p.requires_grad]
		return torch.optim.Adam(params, lr=0.0002)

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
	dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=torchvision.transforms.ToTensor())
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])
	test_dataset = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=torchvision.transforms.ToTensor())

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
	# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html
	print("Training...")
	if True:
		trainer = pl.Trainer(gpus=2, max_epochs=20)
		# Strategy = {"dp", "ddp", "ddp_spawn", "ddp_fork", "ddp_notebook", "horovod", "bagua", "deepspeed"}.
		#	REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", auto_select_gpus=False, max_epochs=20, gradient_clip_val=5, gradient_clip_algorithm="norm")
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=20, gradient_clip_val=5, gradient_clip_algorithm="value")
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy=pl.strategies.DDPStrategy(process_group_backend="gloo", find_unused_parameters=False), auto_select_gpus=False)
		#trainer = pl.Trainer(gpus=1, precision=16)  # FP16 mixed precision.
		#trainer = pl.Trainer(gpus=1, precision="bf16")  # BFloat16 mixed precision.
		#trainer = pl.Trainer(gpus=1, amp_backend="apex", amp_level="O2")  # NVIDIA APEX mixed precision.
		#trainer = pl.Trainer(min_epochs=None, max_epochs=None, min_steps=None, min_steps=-1, max_time=None)
		#trainer = pl.Trainer(default_root_dir="/path/to/checkpoints")  # Saves checkpoints to "/path/to/checkpoints" at every epoch end.
		#trainer = pl.Trainer(resume_from_checkpoint="/path/to/checkpoint.ckpt")  # Resume training. Deprecated.

		#trainer = pl.Trainer(auto_lr_find=False, auto_scale_batch_size=False)
		#trainer = pl.Trainer(auto_lr_find=True)  # Runs learning rate finder, results override hparams.learning_rate.
		#	Runs a learning rate finder algorithm when calling trainer.tune(), to find optimal initial learning rate.
		#trainer = pl.Trainer(auto_scale_batch_size=True)
		#trainer = pl.Trainer(auto_scale_batch_size="binsearch")  # Runs batch size scaling, result overrides hparams.batch_size.
		#	Automatically tries to find the largest batch size that fits into memory, before any training.

		# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html
		#trainer = pl.Trainer(profiler="simple")
		#trainer = pl.Trainer(profiler=pl.profiler.SimpleProfiler())
		#trainer = pl.Trainer(profiler="advanced")
		#trainer = pl.Trainer(profiler=pl.profiler.AdvancedProfiler())
		#trainer = pl.Trainer(profiler="pytorch")
		#trainer = pl.Trainer(profiler=pl.profiler.PyTorchProfiler())
		#trainer = pl.Trainer(profiler="xla")
		#trainer = pl.Trainer(profiler=pl.profiler.XLAProfiler())

		# Tunes hyperparameters before training.
		#trainer.tune(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, datamodule=None, scale_batch_size_kwargs=None, lr_find_kwargs=None)

		# Calls pl.LightningModule.training_step() and pl.LightningModule.validation_step().
		trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
		#trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=None)  # Path/URL of the checkpoint from which training is resumed.
		#trainer.fit(model, datamodule=datamodule)

		# Can be used for retraining, evaluation, and inference after being loaded by pl.LightningModule.load_from_checkpoint() and torch.load().
		#trainer.save_checkpoint("/path/to/checkpoint.ckpt", weights_only=False)  # Pickled object. ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters'].
		# Can be used for evaluation and inference after being loaded by pl.LightningModule.load_from_checkpoint() and torch.load().
		#trainer.save_checkpoint("/path/to/checkpoint.ckpt", weights_only=True)  # Pickled object. ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'hparams_name', 'hyper_parameters'].
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
			dirpath=None,
			filename="{epoch:03d}-{val_acc:.2f}-{val_loss:.2f}",
			monitor="val_loss", mode="min",
			save_top_k=3,
			#save_weights_only=False, save_last=None,
			#every_n_epochs=None, every_n_train_steps=None, train_time_interval=None,
		)
		checkpoint_callback = pl.callbacks.ModelCheckpoint(
			dirpath=None,
			filename="{epoch:03d}-{val_acc:.2f}-{val_loss:.2f}",
			monitor="val_acc", mode="max",
			save_top_k=5,
			#save_weights_only=False, save_last=None,
			#every_n_epochs=None, every_n_train_steps=None, train_time_interval=None,
		)
		"""
		#device_stats_callback = DeviceStatsMonitor()
		#early_stopping_callback = pl.callbacks.EarlyStopping("val_loss", min_delta=0.0, patience=3, verbose=False, mode="min", strict=True, check_finite=True, stopping_threshold=None, divergence_threshold=None, check_on_train_epoch_end=None)
		lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
		#model_pruing_callback = pl.callbacks.ModelPruning(pruning_fn="l1_unstructured", parameters_to_prune=[(model.mlp_1, "weight"), (model.mlp_2, "weight")], parameter_names=None, use_global_unstructured=True, amount=0.5, apply_pruning=True, make_pruning_permanent=True, use_lottery_ticket_hypothesis=True, resample_parameters=False, pruning_dim=None, pruning_norm=None, verbose=0, prune_on_train_epoch_end=True)
		#model_summary_callback = pl.callbacks.ModelSummary(max_depth=1)
		#quantization_aware_training_callback = pl.callbacks.QuantizationAwareTraining(qconfig="fbgemm", observer_type="average", collect_quantization=None, modules_to_fuse=None, input_compatible=True, quantize_on_fit_end=True, observer_enabled_stages=("train",))
		#rich_model_summary_callback = pl.callbacks.RichModelSummary(max_depth=1)
		#rich_progress_bar_callback = pl.callbacks.RichProgressBar(refresh_rate=1, leave=False, theme=RichProgressBarTheme(description="white", progress_bar="#6206E0", progress_bar_finished="#6206E0", progress_bar_pulse="#6206E0", batch_progress="white", time="grey54", processing_speed="grey70", metrics="white"), console_kwargs=None)
		# Stochastic weight averaging (SWA).
		#	When SWA gets activated, SWALR (annealing_strategy) is applied from swa_epoch_start (current LRs) to swa_epoch_start + annealing_epochs (swa_lrs). [swa_epoch_start, swa_epoch_start + annealing_epochs).
		#swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=None, swa_epoch_start=0.8, annealing_epochs=10, annealing_strategy="cos", avg_fn=None, device=None)
		#timer_callback = pl.callbacks.Timer(duration=None, interval=Interval.step, verbose=True)
		#tqdm_progress_bar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=1, process_position=0)

		# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/extensions/plugins.html
		#ddp_plugin = pl.plugins.training_type.DDPPlugin(parallel_devices=None, num_nodes=None, cluster_environment=None, sync_batchnorm=None, ddp_comm_state=None, ddp_comm_hook=None, ddp_comm_wrapper=None)
		#ddp_plugin = pl.plugins.training_type.DDPPlugin(find_unused_parameters=False)
		#amp_plugin = pl.plugins.precision.ApexMixedPrecisionPlugin(amp_level="O2")

		#callbacks = [checkpoint_callback]
		callbacks = [checkpoint_callback, lr_monitor_callback]
		#callbacks = [checkpoint_callback, swa_callback]
		#callbacks = None

		#plugins = [ddp_plugin]
		#plugins = [ddp_plugin, amp_plugin]
		plugins = None

		#tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs")
		#tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs", name="")
		#tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir="./lightning_logs", name="default", version=None, log_graph=False, default_hp_metric=True, prefix="")
		#comet_logger = pl.loggers.CometLogger(save_dir="./comet_logs")

		logger = True
		#logger = tensorboard_logger
		#logger = [tensorboard_logger, comet_logger]

		#trainer = pl.Trainer(devices=-1, accelerator="gpu", auto_select_gpus=False, max_epochs=20, callbacks=None, plugins=None)
		trainer = pl.Trainer(devices=-1, accelerator="gpu", auto_select_gpus=False, max_epochs=20, callbacks=callbacks, enable_checkpointing=True)
		#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=False, plugins=plugins)
		#trainer = pl.Trainer(logger=logger, log_every_n_steps=50)

		trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
		#trainer.fit(model, datamodule=datamodule)

		#best_model_path = trainer.checkpoint_callback.best_model_path
		#best_model_path = checkpoint_callback.best_model_path

		# All init args were saved to the checkpoint.
		checkpoint = torch.load("/path/to/checkpoint.ckpt")
		print("Hyper-parameters: {}.".format(checkpoint["hyper_parameters"]))
	print("The model trained.")

	#--------------------
	#val_metrics = trainer.validate(model=model, dataloaders=dataloader, ckpt_path=None, verbose=True)  # Calls pl.LightningModule.validation_step().
	#test_metrics = trainer.test(model=model, dataloaders=dataloader, ckpt_path=None, verbose=True)  # Calls pl.LightningModule.test_step().
	#predictions = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=None, return_predictions=None)  # Calls pl.LightningModule.predict_step().
	#predictions = model(...)  # Calls pl.LightningModule.forward().

	#trainer.test(model, dataloaders=dataloader, ckpt_path=None)  # Uses the current weights.
	#trainer.test(model, dataloaders=dataloader, ckpt_path="/path/to/checkpoint.ckpt")
	#trainer.test(model, dataloaders=dataloader, ckpt_path="best")  # {"best", "last"}.
	#trainer.test(dataloaders=dataloader, ckpt_path="/path/to/checkpoint.ckpt")
	#trainer.test(dataloaders=dataloader, ckpt_path="best")  # {"best", "last"}.

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
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model.to(device)

		# When using forward(), you are responsible to call eval() and use the no_grad() context manager.
		model.eval()
		model.freeze()
		with torch.no_grad():
			embedding = ...
			reconstruction = model(embedding.to(device)).cpu()
	print("Inferred by the model.")

class LeNet5(pl.LightningModule):
	def __init__(self, num_classes=10):
		super().__init__()
		self.save_hyperparameters()

		self.hparams.batch_size = 0  # For finding the largest batch size that fits into memory when calling trainer.tune().
		self.hparams.learning_rate = 0  # For learning rate finder algorithm when calling trainer.tune().

		# 1 input image channel, 6 output channels, 3x3 square convolution kernel.
		self.conv1 = torch.nn.Conv2d(1, 6, 3)
		self.conv2 = torch.nn.Conv2d(6, 16, 3)
		# An affine operation: y = Wx + b.
		self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # For 28x28 input.
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, num_classes)

		self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")

		for param in self.parameters():
			if param.dim() > 1:
				torch.nn.init.xavier_uniform_(param)  # Initialize parameters with Glorot / fan_avg.

	def on_load_checkpoint(self, checkpoint):
		#self.model = checkpoint["model"]
		pass

	def on_save_checkpoint(self, checkpoint):
		#checkpoint["model"] = self.model
		pass

	def configure_optimizers(self):
		params = [p for p in self.parameters() if p.requires_grad]
		optimizer = torch.optim.Adam(params, lr=1e-2)
		return optimizer

	def forward(self, x):
		# Max pooling over a (2, 2) window.
		x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
		# If the size is a square you can only specify a single number.
		x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
		x = x.view(-1, self._num_flat_features(x))
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def training_step(self, batch, batch_idx):
		"""
		batch_idx  # [0, self.trainer.num_training_batches) = [0, train_steps_per_epoch).

		self.trainer.num_training_batches  # train_steps_per_epoch.
		self.trainer.num_val_batches  # val_steps_per_epoch.
		self.trainer.num_test_batches  # test_steps_per_epoch.
		self.trainer.num_predict_batches
		self.trainer.num_sanity_val_batches

		self.current_epoch  # [0, self.trainer.max_epochs) = [0, num_epochs].
		self.trainer.max_epochs  # num_epochs + 1.
		"""

		start_time = time.time()
		loss, y = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(y, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict(
			{"train_loss": loss, "train_acc": performances["acc"], "train_time": step_time},
			on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True
		)

		return loss

	def validation_step(self, batch, batch_idx):
		start_time = time.time()
		loss, y = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(y, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({"val_loss": loss, "val_acc": performances["acc"], "val_time": step_time}, rank_zero_only=True)

	def test_step(self, batch, batch_idx):
		start_time = time.time()
		loss, y = self._shared_step(batch, batch_idx)
		performances = self._evaluate_performance(y, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict({"test_loss": loss, "test_acc": performances["acc"], "test_time": step_time}, rank_zero_only=True)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		# NOTE [error] >>
		#	pytorch_lightning.utilities.exceptions.MisconfigurationException: You are trying to 'self.log()' but the loop's result collection is not registered yet.
		#	This is most likely because you are trying to log in a 'predict' hook, but it doesn't support logging.
		return self(batch[0])

	def _shared_step(self, batch, batch_idx):
		x, t = batch
		y = self(x)

		loss = self.criterion(y, t)

		return loss, y

	def _evaluate_performance(self, y, batch, batch_idx):
		_, t = batch

		#acc = (torch.argmax(y, dim=-1) == t).sum().item()
		acc = (torch.argmax(y, dim=-1) == t).float().mean().item()

		return {'acc': acc}

	def _num_flat_features(self, x):
		size = x.size()[1:]  # All dimensions except the batch dimension.
		num_features = 1
		for s in size:
			num_features *= s
		return num_features

def lenet5_mnist_example():
	import os
	import numpy as np

	# Prepare data.
	train_transform = torchvision.transforms.ToTensor()
	val_transform = torchvision.transforms.ToTensor()
	train_dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=train_transform, target_transform=None)
	val_dataset = torchvision.datasets.MNIST(root=".", train=False, download=True, transform=val_transform, target_transform=None)

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=4, persistent_workers=False)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=512, num_workers=4, persistent_workers=False)

	#--------------------
	# Training.
	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=None,
		filename="{epoch:03d}-{val_acc:.2f}-{val_loss:.2f}",
		#monitor="val_acc", mode="max",
		monitor="val_loss", mode="min",
		save_top_k=-1,
		save_weights_only=False, save_last=None,
		every_n_epochs=None, every_n_train_steps=None, train_time_interval=None,
	)
	lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
	# NOTE [info] >>
	#	StochasticWeightAveraging is in beta and subject to change.
	#	StochasticWeightAveraging is currently not supported for multiple optimizers/schedulers.
	#	StochasticWeightAveraging is currently only supported on every epoch.
	#	When SWA gets activated, SWALR (annealing_strategy) is applied from swa_epoch_start (current LRs) to swa_epoch_start + annealing_epochs (swa_lrs). [swa_epoch_start, swa_epoch_start + annealing_epochs).
	#swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1.0e-5, swa_epoch_start=0.5, annealing_epochs=5, annealing_strategy="cos", avg_fn=None)
	swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1.0e-5, swa_epoch_start=10, annealing_epochs=5, annealing_strategy="cos", avg_fn=None, device=None)
	callbacks = [checkpoint_callback, lr_monitor_callback, swa_callback]

	profiler = "pytorch"  # {None, "simple", "advanced", "pytorch", "xla"}.
	#profiler = None

	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy=None, auto_select_gpus=True, max_epochs=-1, precision=16)
	trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=20, precision=16, callbacks=callbacks, profiler=profiler)
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=10, precision=16, limit_train_batches=1.0, limit_test_batches=1.0)
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=10, precision=16, auto_lr_find=True)
	# NOTE [error] >> pytorch_lightning.utilities.exceptions.MisconfigurationException: The batch scaling feature cannot be used with dataloaders passed directly to '.fit()'. Please disable the feature or incorporate the dataloader into the model.
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=10, precision=16, auto_scale_batch_size="binsearch", auto_lr_find=True)

	if True:
		# Case 1: Training.

		# Build a model.
		model = LeNet5(num_classes=10)

		# Tune hyperparameters before training.
		if True:
			tuning_results = trainer.tune(
				model,
				train_dataloaders=train_dataloader,
				val_dataloaders=val_dataloader,
				datamodule=None,
				scale_batch_size_kwargs={
					"mode": "power",
					"steps_per_trial": 3,
					"init_val": 2,
					"max_trials": 25,
					"batch_arg_name": "batch_size",
				},
				lr_find_kwargs={
					"min_lr": 1e-6,
					"max_lr": 10.0,
					"num_training": 100,
					"mode": "exponential",
					"early_stop_threshold": 4.0,
					"update_attr": False,
				},
			)
			print(f"Tuning results: {tuning_results}.")
		else:
			# Find optimal learning rate.
			lr_results = trainer.tuner.lr_find(  # pl.tuner.tuning.Tuner class.
				model,
				train_dataloaders=train_dataloader,
				val_dataloaders=val_dataloader,
				datamodule=None,
				min_lr=1e-6,
				max_lr=10.0,
				num_training=100,
				mode="exponential",
				early_stop_threshold=4.0,
				update_attr=False,
			)

			print(f"Suggested learning rate: {lr_results.suggestion()}.")
			fig = lr_results.plot(show=True, suggest=True)
			fig.show()

			# Iteratively try to find the largest batch size for a given model that does not give an out of memory (OOM) error.
			# FIXME [error] >>
			#	pytorch_lightning.utilities.exceptions.MisconfigurationException: The batch scaling feature cannot be used with dataloaders passed directly to '.fit()'.
			#	Please disable the feature or incorporate the dataloader into the model.
			max_batch_size = trainer.tuner.scale_batch_size(  # pl.tuner.tuning.Tuner class.
				model,
				train_dataloaders=train_dataloader,
				val_dataloaders=val_dataloader,
				datamodule=None,
				mode="power",
				steps_per_trial=3,
				init_val=2,
				max_trials=25,
				batch_arg_name="batch_size",
			)

			print(f"Suggested batch size: {max_batch_size}.")

		# Train the model.
		trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

		model_filepath = trainer.checkpoint_callback.best_model_path
		print("The best trained model saved to {}.".format(model_filepath))
	else:
		# Case 2: Loading.

		model_filepath = "./lightning_logs/version_0/checkpoints/epoch=9-step=1180.ckpt"
		assert os.path.isfile(model_filepath), "Model file not found, {}".format(model_filepath)

		# Load a model.
		model = LeNet5.load_from_checkpoint(model_filepath)
		#model = LeNet5.load_from_checkpoint(model_filepath, map_location={"cuda:1": "cuda:0"})

		print("A trained model loaded from {}.".format(model_filepath))

	#--------------------
	if True:
		# Working for Case 1 & 2.

		val_metrics = trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=None, verbose=True)
		print("Validation metrics: {}.".format(val_metrics))

		test_metrics = trainer.test(model=model, dataloaders=val_dataloader, ckpt_path=None, verbose=True)
		print("Test metrics: {}.".format(test_metrics))

		predictions = trainer.predict(model=model, dataloaders=val_dataloader, ckpt_path=None, return_predictions=None)  # A list of [batch size, #classes]'s.
		predictions = torch.vstack(predictions)
		#predictions = torch.argmax(torch.vstack(predictions), dim=-1)
		print("Predictions: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, torch.min(predictions), torch.max(predictions)))
	else:
		# Working for Case 1.

		# NOTE [error] >> For Case 2.
		#	AttributeError: 'NoneType' object has no attribute 'cpu'.
		#		self.lightning_module.cpu()

		val_metrics = trainer.validate(model=None, dataloaders=val_dataloader, ckpt_path=model_filepath, verbose=True)
		print("Validation metrics: {}.".format(val_metrics))

		test_metrics = trainer.test(model=None, dataloaders=val_dataloader, ckpt_path=model_filepath, verbose=True)
		print("Test metrics: {}.".format(test_metrics))

		predictions = trainer.predict(model=None, dataloaders=val_dataloader, ckpt_path=model_filepath, return_predictions=None)  # A list of [batch size, #classes]'s.
		predictions = torch.vstack(predictions)
		#predictions = torch.argmax(torch.vstack(predictions), dim=-1)
		print("Predictions: shape = {}, dtype = {}, (min, max) = ({}, {}).".format(predictions.shape, predictions.dtype, torch.min(predictions), torch.max(predictions)))

	#--------------------
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = model.to(device)
	model.eval()
	model.freeze()

	gts, predictions = list(), list()
	with torch.no_grad():
		for batch_inputs, batch_outputs in val_dataloader:
			gts.append(batch_outputs.numpy())  # [batch size].
			predictions.append(model(batch_inputs.to(device)).cpu().numpy())  # [batch size, #classes].
	gts, predictions = np.hstack(gts), np.argmax(np.vstack(predictions), axis=-1)
	assert len(gts) == len(predictions)
	num_examples = len(gts)

	results = gts == predictions
	num_correct_examples = results.sum().item()
	acc = results.mean().item()

	print("Prediction: accuracy = {} / {} = {}.".format(num_correct_examples, num_examples, acc))

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
		params = [p for p in self.parameters() if p.requires_grad]
		optimizer = torch.optim.Adam(params, lr=1e-3)
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
	dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=torchvision.transforms.ToTensor())
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
	#trainer = pl.Trainer(gpus=2, num_nodes=1, precision=16, limit_train_batches=0.5, accelerator="dp", max_epochs=10)

	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

class Seq2SeqModule(pl.LightningModule):
	def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, tgt_pad_idx: int):
		super().__init__()
		self.save_hyperparameters(ignore=["encoder", "decoder"])

		self.model = torch.nn.Sequential(
			encoder,
			decoder
		)

		self.criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

		# Initialize parameters.
		if False:
			for name, param in self.model.named_parameters():
				if "weight" in name:
					torch.nn.init.normal_(param.data, mean=0, std=0.01)
				else:
					torch.nn.init.constant_(param.data, 0)
		else:
			def init_weights(m: torch.nn.Module) -> None:
				for name, param in m.named_parameters():
					if "weight" in name:
						torch.nn.init.normal_(param.data, mean=0, std=0.01)
					else:
						torch.nn.init.constant_(param.data, 0)

			self.model.apply(init_weights)

		print(f"The model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters.")

	def configure_optimizers(self) -> pl.utilities.types.OptimizerLRScheduler:
		#optimizer = torch.optim.Adam(self.model.parameters())
		params = [p for p in self.model.parameters() if p.requires_grad]
		optimizer = torch.optim.Adam(params)
		return optimizer

	def forward(self, x: torch.Tensor, max_len: int, tgt_sos_idx: int, tgt_eos_idx: int) -> torch.Tensor:
		encoder, decoder = self.model[0], self.model[1]

		encoder_outputs, hidden = encoder(x)

		batch_size = x.shape[1]
		tgt_vocab_size = decoder.output_dim
		outputs = torch.zeros(max_len, batch_size, tgt_vocab_size, device=self.device)
		output = torch.full((batch_size,), fill_value=tgt_sos_idx, device=self.device)  # First input to the decoder is the <sos> token.
		for t in range(1, max_len):
			output, hidden = decoder(output, hidden, encoder_outputs)
			outputs[t] = output
			output = output.max(1)[1]

		return outputs

	def training_step(self, batch: typing.Any, batch_idx: typing.Any) -> typing.Any:
		start_time = time.time()
		loss, ppl = self._shared_step(batch, batch_idx, teacher_forcing_ratio=0.5)
		step_time = time.time() - start_time

		self.log_dict(
			{"train_loss": loss, "train_ppl": ppl, "train_time": step_time},
			on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True, batch_size=batch[0].shape[1]
		)

		return loss

	def validation_step(self, batch: typing.Any, batch_idx: typing.Any) -> None:
		start_time = time.time()
		loss, ppl = self._shared_step(batch, batch_idx, teacher_forcing_ratio=0)  # Turn off teacher forcing.
		step_time = time.time() - start_time

		self.log_dict({"val_loss": loss, "val_ppl": ppl, "val_time": step_time}, rank_zero_only=True, sync_dist=True, batch_size=batch[0].shape[1])

	def test_step(self, batch: typing.Any, batch_idx: typing.Any) -> None:
		start_time = time.time()
		loss, ppl = self._shared_step(batch, batch_idx, teacher_forcing_ratio=0)  # Turn off teacher forcing.
		step_time = time.time() - start_time

		self.log_dict({"test_loss": loss, "test_ppl": ppl, "test_time": step_time}, rank_zero_only=True, sync_dist=True, batch_size=batch[0].shape[1])

	def predict_step(self, batch: typing.Any, batch_idx: typing.Any, dataloader_idx: typing.Any = None) -> typing.Any:
		raise NotImplementedError

	def _shared_step(self, batch: typing.Any, batch_idx: typing.Any, teacher_forcing_ratio: float) -> typing.Any:
		src, tgt = batch
		encoder, decoder = self.model[0], self.model[1]

		encoder_outputs, hidden = encoder(src)

		batch_size = src.shape[1]
		max_len = tgt.shape[0]
		tgt_vocab_size = decoder.output_dim
		outputs = torch.zeros(max_len, batch_size, tgt_vocab_size, device=self.device)
		output = tgt[0,:]  # First input to the decoder is the <sos> token.
		for t in range(1, max_len):
			output, hidden = decoder(output, hidden, encoder_outputs)
			outputs[t] = output
			teacher_force = random.random() < teacher_forcing_ratio
			top1 = output.max(1)[1]
			output = (tgt[t] if teacher_force else top1)

		outputs = outputs[1:].view(-1, outputs.shape[-1])
		tgt = tgt[1:].view(-1)

		loss = self.criterion(outputs, tgt)
		ppl = math.exp(loss)

		return loss, ppl

# REF [function] >> torchtext_translation_tutorial() in ../pytorch/pytorch_neural_network.py
def torchtext_translation_tutorial():
	import io
	from collections import Counter
	import numpy as np
	import torchtext

	# Download the raw data for the English and German Spacy tokenizers.
	#	python -m spacy download en
	#	python -m spacy download de

	#--------------------
	# Data processing.

	url_base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
	train_urls = ("train.de.gz", "train.en.gz")
	val_urls = ("val.de.gz", "val.en.gz")
	test_urls = ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz")

	train_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in train_urls]
	val_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in val_urls]
	test_filepaths = [torchtext.utils.extract_archive(torchtext.utils.download_from_url(url_base + url))[0] for url in test_urls]

	de_tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="de")
	en_tokenizer = torchtext.data.utils.get_tokenizer("spacy", language="en")

	def build_vocab(filepath, tokenizer):
		counter = Counter()
		with io.open(filepath, encoding="utf8") as f:
			for string_ in f:
				counter.update(tokenizer(string_))
		return torchtext.vocab.vocab(counter, specials=["<unk>", "<pad>", "<bos>", "<eos>"])

	de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
	en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

	def data_process(filepaths):
		raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
		raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
		data = []
		for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
			de_tensor_ = torch.tensor([de_vocab[token if token in de_vocab else "<unk>"] for token in de_tokenizer(raw_de)], dtype=torch.long)
			en_tensor_ = torch.tensor([en_vocab[token if token in en_vocab else "<unk>"] for token in en_tokenizer(raw_en)], dtype=torch.long)
			data.append((de_tensor_, en_tensor_))
		return data

	train_dataset = data_process(train_filepaths)
	val_dataset = data_process(val_filepaths)
	test_dataset = data_process(test_filepaths)

	print(f"#train data = {len(train_dataset)}, #validation data = {len(val_dataset)}, #test data = {len(test_dataset)}.")

	PAD_IDX = de_vocab["<pad>"]
	BOS_IDX = de_vocab["<bos>"]
	EOS_IDX = de_vocab["<eos>"]
	assert PAD_IDX == en_vocab["<pad>"]
	assert BOS_IDX == en_vocab["<bos>"]
	assert EOS_IDX == en_vocab["<eos>"]

	def generate_batch(data_batch):
		de_batch, en_batch = [], []
		for (de_item, en_item) in data_batch:
			de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
			en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
		de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=PAD_IDX)
		en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=PAD_IDX)
		return de_batch, en_batch

	BATCH_SIZE = 128
	num_workers = 8
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, collate_fn=generate_batch)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=generate_batch)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, collate_fn=generate_batch)

	print(f"#train steps per epoch = {len(train_dataloader)}, #validation steps per epoch = {len(val_dataloader)}, #test steps per epoch = {len(test_dataloader)}.")

	#--------------------
	# Defining our module.

	class Encoder(torch.nn.Module):
		def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float):
			super().__init__()

			self.input_dim = input_dim
			self.emb_dim = emb_dim
			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim
			self.dropout = dropout

			self.embedding = torch.nn.Embedding(input_dim, emb_dim)
			self.rnn = torch.nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
			self.fc = torch.nn.Linear(enc_hid_dim * 2, dec_hid_dim)
			self.dropout = torch.nn.Dropout(dropout)

		def forward(self, src: torch.Tensor) -> typing.Tuple[torch.Tensor]:
			embedded = self.dropout(self.embedding(src))
			outputs, hidden = self.rnn(embedded)
			hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
			return outputs, hidden

	class Attention(torch.nn.Module):
		def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
			super().__init__()

			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim

			self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
			self.attn = torch.nn.Linear(self.attn_in, attn_dim)

		def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
			src_len = encoder_outputs.shape[0]
			repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
			encoder_outputs = encoder_outputs.permute(1, 0, 2)
			energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
			attention = torch.sum(energy, dim=2)
			return torch.nn.functional.softmax(attention, dim=1)

	class Decoder(torch.nn.Module):
		def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: int, attention: torch.nn.Module):
			super().__init__()

			self.emb_dim = emb_dim
			self.enc_hid_dim = enc_hid_dim
			self.dec_hid_dim = dec_hid_dim
			self.output_dim = output_dim
			self.dropout = dropout
			self.attention = attention

			self.embedding = torch.nn.Embedding(output_dim, emb_dim)
			self.rnn = torch.nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
			self.out = torch.nn.Linear(self.attention.attn_in + emb_dim, output_dim)
			self.dropout = torch.nn.Dropout(dropout)

		def _weighted_encoder_rep(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
			a = self.attention(decoder_hidden, encoder_outputs)
			a = a.unsqueeze(1)

			encoder_outputs = encoder_outputs.permute(1, 0, 2)
			weighted_encoder_rep = torch.bmm(a, encoder_outputs)
			weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
			return weighted_encoder_rep

		def forward(self, input: torch.Tensor, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> typing.Tuple[torch.Tensor]:
			input = input.unsqueeze(0)
			embedded = self.dropout(self.embedding(input))
			weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
			rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
			output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
			embedded = embedded.squeeze(0)
			output = output.squeeze(0)
			weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
			output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))
			return output, decoder_hidden.squeeze(0)

	INPUT_DIM = len(de_vocab)
	OUTPUT_DIM = len(en_vocab)
	#ENC_EMB_DIM = 256
	#DEC_EMB_DIM = 256
	#ENC_HID_DIM = 512
	#DEC_HID_DIM = 512
	#ATTN_DIM = 64
	ENC_EMB_DIM = 32
	DEC_EMB_DIM = 32
	ENC_HID_DIM = 64
	DEC_HID_DIM = 64
	ATTN_DIM = 8
	ENC_DROPOUT = 0.5
	DEC_DROPOUT = 0.5

	enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
	attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
	dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

	#PAD_IDX = en_vocab["<pad>"]
	model = Seq2SeqModule(enc, dec, PAD_IDX)

	#--------------------
	# Training the Seq2Seq model.

	checkpoint_callback = pl.callbacks.ModelCheckpoint(
		dirpath=None,
		filename="{epoch:03d}-{val_ppl:.2f}-{val_loss:.2f}",
		#monitor="val_ppl", mode="min",
		monitor="val_loss", mode="min",
		save_top_k=-1,
		save_weights_only=False, save_last=None,
		every_n_epochs=None, every_n_train_steps=None, train_time_interval=None,
	)
	lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False)
	pl_callbacks = [checkpoint_callback, lr_monitor_callback]

	num_epochs = 10
	# RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed. You can make a clone to get a normal tensor before doing inplace update. See https://github.com/pytorch/rfcs/pull/17 for more details.
	#trainer = pl.Trainer(devices=-1, accelerator="gpu", strategy="dp", auto_select_gpus=True, max_epochs=num_epochs, gradient_clip_val=1, gradient_clip_algorithm="norm", callbacks=pl_callbacks)
	#trainer = pl.Trainer(devices="auto", accelerator="gpu", strategy="auto", precision="32-true", max_epochs=num_epochs, gradient_clip_val=1, gradient_clip_algorithm="norm", callbacks=pl_callbacks)
	trainer = pl.Trainer(devices=1, accelerator="gpu", strategy="auto", precision="16-mixed", max_epochs=num_epochs, gradient_clip_val=1, gradient_clip_algorithm="norm", callbacks=pl_callbacks)

	print("Training...")
	start_time = time.time()
	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
	print(f"Trained: {time.time() - start_time} secs.")

	model_filepath = trainer.checkpoint_callback.best_model_path
	print(f"The best trained model saved to {model_filepath}.")

	#-----
	# INFO [error] >>
	#	<error> FileNotFoundError: Checkpoint file not found.
	#	<env> When using pl.Trainer(devices="auto", accelerator="gpu", strategy="auto", ...)
	#	<cause> Checkpoint files are overwritten by another process.

	print("Validating...")
	start_time = time.time()
	val_metrics = trainer.validate(dataloaders=val_dataloader, ckpt_path="best", verbose=True)
	print(f"Validated: {time.time() - start_time} secs.")
	print(f"Validation metrics: {val_metrics}.")

	print("Testing...")
	start_time = time.time()
	test_metrics = trainer.test(dataloaders=test_dataloader, ckpt_path="best", verbose=True)
	print(f"Tested: {time.time() - start_time} secs.")
	print(f"Test metrics: {test_metrics}.")

	#-----
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}.")

	model = model.to(device)
	model.eval()
	model.freeze()

	if False:
		# Not working.

		print("Predicting...")
		start_time = time.time()
		num_examples, num_correct_examples = 0, 0
		with torch.no_grad():
			for batch_inputs, batch_outputs in test_dataloader:
				predictions = model(batch_inputs.to(device), max_len=batch_outputs.shape[0], tgt_sos_idx=BOS_IDX, tgt_eos_idx=EOS_IDX)
				predictions = np.argmax(predictions.cpu().numpy(), axis=-1)  # [time-steps, batch size, #classes] -> [time-steps, batch size].

				gts = np.transpose(batch_outputs.numpy(), (1, 0))  # [time-steps, batch size] -> [batch size, time-steps].
				predictions = np.transpose(predictions, (1, 0))  # [time-steps, batch size] -> [batch size, time-steps].
				assert len(gts) == len(predictions)
				num_examples += len(gts)
				num_correct_examples += sum([np.all(gt == pred) for gt, pred in zip(gts, predictions)])  # TODO [modify] >> Not good.
		print(f"Predicted: {time.time() - start_time} secs.")
		print(f"Prediction: accuracy = {num_correct_examples} / {num_examples} = {num_correct_examples / num_examples}.")
	else:
		print("Predicting...")
		start_time = time.time()
		criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
		ppl, num_examples = 0.0, 0
		with torch.no_grad():
			for batch_inputs, batch_outputs in test_dataloader:
				predictions = model(batch_inputs.to(device), max_len=batch_outputs.shape[0], tgt_sos_idx=BOS_IDX, tgt_eos_idx=EOS_IDX)

				predictions = predictions.cpu()
				predictions = predictions[1:].view(-1, predictions.shape[-1])
				batch_outputs = batch_outputs[1:].view(-1)

				loss = criterion(predictions, batch_outputs)
				ppl += math.exp(loss.cpu().item()) * predictions.shape[1]
				num_examples += predictions.shape[1]
		print(f"Predicted: {time.time() - start_time} secs.")
		print(f"Prediction: PPL = {ppl / num_examples}.")

class Lang:
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS.

	def addSentence(self, sentence):
		for word in sentence.split(" "):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

class LangDataset(torch.utils.data.Dataset):
	def __init__(self, sos, eos, max_length):
		super().__init__()

		self.sos, self.eos = sos, eos
		self.max_length = max_length

		eng_prefixes = (
			"i am ", "i m ",
			"he is", "he s ",
			"she is", "she s ",
			"you are", "you re ",
			"we are", "we re ",
			"they are", "they re "
		)

		self.input_lang, self.output_lang, self.pairs = self.prepareData("eng", "fra", eng_prefixes, True)
		print(random.choice(self.pairs))

		self.pairs = [self.tensorsFromPair(pair) for pair in self.pairs]

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		return self.pairs[idx]

	def prepareData(self, lang1, lang2, eng_prefixes, reverse=False):
		input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
		print("Read %s sentence pairs" % len(pairs))
		pairs = self.filterPairs(pairs, eng_prefixes)
		print("Trimmed to %s sentence pairs" % len(pairs))
		print("Counting words...")
		for pair in pairs:
			input_lang.addSentence(pair[0])
			output_lang.addSentence(pair[1])
		print("Counted words:")
		print(input_lang.name, input_lang.n_words)
		print(output_lang.name, output_lang.n_words)
		return input_lang, output_lang, pairs

	@staticmethod
	def readLangs(lang1, lang2, reverse=False):
		print("Reading lines...")

		# Read the file and split into lines.
		lines = open("%s-%s.txt" % (lang1, lang2), encoding="utf-8").read().strip().split("\n")

		# Split every line into pairs and normalize.
		pairs = [[LangDataset.normalizeString(s) for s in l.split("\t")] for l in lines]

		# Reverse pairs, make Lang instances.
		if reverse:
			pairs = [list(reversed(p)) for p in pairs]
			input_lang = Lang(lang2)
			output_lang = Lang(lang1)
		else:
			input_lang = Lang(lang1)
			output_lang = Lang(lang2)

		return input_lang, output_lang, pairs

	# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427.
	@staticmethod
	def unicodeToAscii(s):
		return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

	# Lowercase, trim, and remove non-letter characters.
	@staticmethod
	def normalizeString(s):
		s = LangDataset.unicodeToAscii(s.lower().strip())
		s = re.sub(r"([.!?])", r" \1", s)
		s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
		return s

	def filterPair(self, p, eng_prefixes):
		return len(p[0].split(" ")) < self.max_length and len(p[1].split(" ")) < self.max_length and p[1].startswith(eng_prefixes)

	def filterPairs(self, pairs, eng_prefixes):
		return [pair for pair in pairs if self.filterPair(pair, eng_prefixes)]

	@staticmethod
	def indexesFromSentence(lang, sentence):
		return [lang.word2index[word] for word in sentence.split(" ")]

	def tensorFromSentence(self, lang, sentence):
		indexes = LangDataset.indexesFromSentence(lang, sentence)
		indexes.append(self.eos)
		return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

	def tensorsFromPair(self, pair):
		input_tensor = self.tensorFromSentence(self.input_lang, pair[0])
		target_tensor = self.tensorFromSentence(self.output_lang, pair[1])
		return (input_tensor, target_tensor)

class EncoderRNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size):
		super().__init__()
		self.hidden_size = hidden_size

		self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
		self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self, device):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(torch.nn.Module):
	def __init__(self, hidden_size, output_size):
		super().__init__()

		self.hidden_size = hidden_size

		self.embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
		self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
		self.out = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
		self.softmax = torch.nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		output = self.embedding(input).view(1, 1, -1)
		output = torch.nn.functional.relu(output)
		output, hidden = self.gru(output, hidden)
		output = self.softmax(self.out(output[0]))
		return output, hidden

	def initHidden(self, device):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class AttentionDecoderRNN(torch.nn.Module):
	def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
		super().__init__()

		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_p = dropout_p
		self.max_length = max_length

		self.embedding = torch.nn.Embedding(num_embeddings=output_size, embedding_dim=hidden_size)
		self.attn = torch.nn.Linear(in_features=hidden_size * 2, out_features=max_length, bias=True)
		self.attn_combine = torch.nn.Linear(in_features=hidden_size * 2, out_features=hidden_size, bias=True)
		self.dropout = torch.nn.Dropout(p=dropout_p, inplace=False)
		self.gru = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False)
		self.out = torch.nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)

		attn_weights = torch.nn.functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)

		output = torch.nn.functional.relu(output)
		output, hidden = self.gru(output, hidden)

		output = torch.nn.functional.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def initHidden(self, device):
		return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderDecoderModel(pl.LightningModule):
	def __init__(self, input_dim, output_dim, hidden_dim, sos, eos, max_length, teacher_forcing_ratio):
		super().__init__()

		self.input_dim = input_dim
		self.output_dim = output_dim
		self.sos, self.eos = sos, eos
		self.max_length = max_length
		self.teacher_forcing_ratio = teacher_forcing_ratio

		self.encoder = EncoderRNN(input_dim, hidden_dim)
		#self.decoder = DecoderRNN(hidden_dim, output_dim)
		self.decoder = AttentionDecoderRNN(hidden_dim, output_dim, dropout_p=0.1)

		self.criterion = torch.nn.NLLLoss()

		# Initialize parameters with Glorot / fan_avg.
		for p in self.encoder.parameters():
			if p.dim() > 1:
				torch.nn.init.xavier_uniform_(p)
		for p in self.decoder.parameters():
			if p.dim() > 1:
				torch.nn.init.xavier_uniform_(p)

	def configure_optimizers(self):
		params = [p for p in self.parameters() if p.requires_grad]
		optimizer = torch.optim.Adam(params, lr=0.0001)
		return optimizer

	def forward(self, x):
		encoder_hidden = self.encoder.initHidden(self.device)  # Set hiddens to initial values.

		input_length = x.size(0)

		encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)
		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[self.sos]], device=self.device)  # SOS.
		decoder_hidden = encoder_hidden

		decoded_words = []
		#decoder_attentions = torch.zeros(self.max_length, self.max_length, device=self.device)
		for di in range(self.max_length):
			decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
			#decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == self.eos:
				#decoded_words.append("<EOS>")
				break
			else:
				#decoded_words.append(output_lang.index2word[topi.item()])
				decoded_words.append(topi.item())

			decoder_input = topi.squeeze().detach()

		#return decoded_words, decoder_attentions[:di + 1]
		return decoded_words

	def training_step(self, batch, batch_idx):
		start_time = time.time()
		x, t = batch
		assert len(x) == 1 and len(t) == 1
		x, t = x[0], t[0]

		encoder_hidden = self.encoder.initHidden(self.device)  # Set hiddens to initial values.

		input_length = x.size(0)
		target_length = t.size(0)

		encoder_outputs = torch.zeros((self.max_length, self.encoder.hidden_size), device=self.device)
		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[self.sos]], device=self.device)
		decoder_hidden = encoder_hidden

		use_teacher_forcing = random.random() < self.teacher_forcing_ratio

		loss = 0
		if use_teacher_forcing:
			# Teacher forcing: feed the target as the next input.
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				loss += self.criterion(decoder_output, t[di])
				decoder_input = t[di]  # Teacher forcing.
		else:
			# Without teacher forcing: use its own predictions as the next input.
			for di in range(target_length):
				decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # Detach from history as input.

				loss += self.criterion(decoder_output, t[di])
				if decoder_input.item() == self.eos:
					break

		#performances = self._evaluate_performance(decoder_output, batch, batch_idx)
		step_time = time.time() - start_time

		self.log_dict(
			#{"train_loss": loss, "train_acc": performances["acc"], "train_time": step_time},
			{"train_loss": loss, "train_time": step_time},
			on_step=True, on_epoch=True, prog_bar=True, logger=True, rank_zero_only=True
		)

		return loss

	def validation_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		#performances = self._evaluate_performance(y, batch, batch_idx)
		step_time = time.time() - start_time

		#self.log_dict({"val_loss": loss, "val_acc": performances["acc"], "val_time": step_time}, rank_zero_only=True)
		self.log_dict({"val_loss": loss, "val_time": step_time}, rank_zero_only=True)

	def test_step(self, batch, batch_idx):
		start_time = time.time()
		loss = self._shared_step(batch, batch_idx)
		#performances = self._evaluate_performance(y, batch, batch_idx)
		step_time = time.time() - start_time

		#self.log_dict({"test_loss": loss, "test_acc": performances["acc"], "test_time": step_time}, rank_zero_only=True)
		self.log_dict({"test_loss": loss, "test_time": step_time}, rank_zero_only=True)

	def _shared_step(self, batch, batch_idx):
		x, t = batch
		assert len(x) == 1 and len(t) == 1
		x, t = x[0], t[0]

		encoder_hidden = self.encoder.initHidden(self.device)  # Set hiddens to initial values.

		input_length = len(x)
		target_length = len(t)

		encoder_outputs = torch.zeros((self.max_length, self.encoder.hidden_size), device=self.device)
		for ei in range(input_length):
			encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
			encoder_outputs[ei] = encoder_output[0, 0]

		decoder_input = torch.tensor([[self.sos]], device=self.device)
		decoder_hidden = encoder_hidden

		loss = 0
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # Detach from history as input.

			loss += self.criterion(decoder_output, t[di])
			if decoder_input.item() == self.eos:
				break

		return loss

	def _evaluate_performance(self, y, batch, batch_idx):
		_, t = batch
		assert len(y) == 1 and len(t) == 1
		y, t = y[0], t[0]

		acc = difflib.SequenceMatcher(None, y.squeeze(), t.squeeze()).ratio()  # [0, 1].

		return {'acc': acc}

def collate_fn(batch):
	return list(zip(*batch))

# REF [site] >> https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
def seq2seq_translation_tutorial():
	SOS_token = 0  # Fixed.
	EOS_token = 1  # Fixed.
	MAX_LENGTH = 10
	teacher_forcing_ratio = 0.5
	hidden_size = 256
	num_epochs = 20
	batch_size = 1  # Fixed.
	num_workers = 8
	train_test_ratio= 0.9

	# Data.
	dataset = LangDataset(SOS_token, EOS_token, max_length=MAX_LENGTH)

	print(f"len(dataset) = {len(dataset)}.")
	print(f"Input dim = {dataset.input_lang.n_words}, output dim = {dataset.output_lang.n_words}.")

	num_examples = len(dataset)
	num_train_examples = int(num_examples * train_test_ratio)
	train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train_examples, num_examples - num_train_examples])
	train_dataset, val_dataset = dataset[:num_train_examples], dataset[num_train_examples:]

	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, collate_fn=collate_fn)
	val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=collate_fn)

	print(f"#train data = {len(train_dataset)}, #validation data = {len(val_dataset)}.")
	print(f"#train steps = {len(train_dataloader)}, #validation steps = {len(val_dataloader)}.")

	#--------------------
	# Model.
	model = EncoderDecoderModel(dataset.input_lang.n_words, dataset.output_lang.n_words, hidden_size, SOS_token, EOS_token, MAX_LENGTH, teacher_forcing_ratio)

	#--------------------
	# Training.
	trainer = pl.Trainer(devices=-1, accelerator="gpu", auto_select_gpus=False, max_epochs=num_epochs, enable_checkpointing=True)

	trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/03-initialization-and-optimization.html
def initalization_and_optimization_tutorial():
	import os, copy, json
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
		loss = torch.nn.functional.cross_entropy(preds, labels)  # Same as nn.CrossEntropyLoss, but as a function instead of module.
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
		torch.optim = optimizer_func([weights])

		list_points = []
		for _ in range(num_updates):
			loss = curve_func(weights[0], weights[1])
			list_points.append(torch.cat([weights.data.detach(), loss.unsqueeze(dim=0).detach()], dim=0))
			torch.optim.zero_grad()
			loss.backward()
			torch.optim.step()
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
			params = [p for p in self.parameters() if p.requires_grad]
			# We will support Adam or SGD as optimizers.
			if self.hparams.optimizer_name == "Adam":
				# AdamW is Adam with a correct implementation of weight decay (see here
				# for details: https://arxiv.org/pdf/1711.05101.pdf).
				optimizer = optim.AdamW(params, **self.hparams.optimizer_hparams)
			elif self.hparams.optimizer_name == "SGD":
				optimizer = optim.SGD(params, **self.hparams.optimizer_hparams)
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

# REF [site] >> https://pytorch-lightning.readthedocs.io/en/stable/cli/lightning_cli_intermediate.html
def lightning_cli_example():
	raise NotImplementedError

def main():
	#minimal_example()

	#lenet5_mnist_example()
	#simple_autoencoder_example()

	torchtext_translation_tutorial()  # Seq2seq model.
	#seq2seq_translation_tutorial()  # Not good.

	#initalization_and_optimization_tutorial()
	#inception_resnet_densenet_tutorial()

	#lightning_cli_example()  # Not yet implemented.

	#-----
	# Port PyTorch Lightning implementation to PyTorch implementation.
	#	Refer to lenet5_mnist_test() in ${SWDT_PYTHON_HOME}/rnd/test/machine_learning/pytorch/pytorch_neural_network.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
