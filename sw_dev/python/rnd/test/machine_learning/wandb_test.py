#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import wandb

# REF [site] >> https://github.com/wandb/wandb
def quickstart():
	"""
	# Start a W&B Run with wandb.init.
	run = wandb.init(project="my_first_project")

	# Save model inputs and hyperparameters in a wandb.config object.
	config = run.config
	config.learning_rate = 0.01
	#config.epochs = 4
	#config.batch_size = 32
	#config.architecture = "resnet"

	# Model training code here ...

	# Log metrics over time to visualize performance with wandb.log.
	for i in range(10):
		run.log({"loss": loss})
	"""

	raise NotImplementedError

# REF [site] >> https://github.com/wandb/examples
def simple_integration_with_framework_example():
	"""
	# 1. Start a W&B run.
	wandb.init(project="gpt3")

	# 2. Save model inputs and hyperparameters.
	config = wandb.config
	config.learning_rate = 0.01

	# Model training code here ...

	# 3. Log metrics over time to visualize performance.
	for i in range (10):
		wandb.log({"loss": loss})
	"""

	raise NotImplementedError
		
# REF [site] >> https://github.com/wandb/wandb
def sklearn_example():
	import sklearn.datasets, sklearn.ensemble, sklearn.model_selection
	from wandb.sklearn import plot_precision_recall, plot_feature_importances
	from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

	# Load and process data.
	wbcd = sklearn.datasets.load_breast_cancer()
	feature_names = wbcd.feature_names
	labels = wbcd.target_names

	test_size = 0.2
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(wbcd.data, wbcd.target, test_size=test_size)

	# Train a model.
	model = sklearn.ensemble.RandomForestClassifier()
	model.fit(X_train, y_train)
	model_params = model.get_params()

	# Get predictions.
	y_pred = model.predict(X_test)
	y_probas = model.predict_proba(X_test)
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]

	#-----
	# Start a new wandb run and add your model hyperparameters.
	run = wandb.init(project="my-awesome-project", config=model_params)

	# Add additional configs to wandb.
	run.config.update({
		"test_size": test_size,
		"train_len": len(X_train),
		"test_len": len(X_test),
	})

	# Log additional visualisations to wandb.
	plot_class_proportions(y_train, y_test, labels)
	plot_learning_curve(model, X_train, y_train)
	plot_roc(y_test, y_probas, labels)
	plot_precision_recall(y_test, y_probas, labels)
	plot_feature_importances(model)

	# [optional] finish the wandb run, necessary in notebooks.
	run.finish()

# REF [site] >> https://github.com/wandb/examples
def pytorch_example():
	"""
	# 1. Start a new run.
	wandb.init(project="gpt-3")

	# 2. Save model inputs and hyperparameters.
	config = wandb.config
	config.dropout = 0.01

	# 3. Log gradients and model parameters.
	wandb.watch(model)
	for batch_idx, (data, target) in enumerate(train_loader):
		#...
		if batch_idx % log_interval == 0:
			# 4. Log metrics to visualize performance.
			wandb.log({"loss": loss})
	"""

	raise NotImplementedError

# REF [site] >> https://github.com/wandb/wandb
def pytorch_lightning_example():
	"""
	from pytorch_lightning.loggers import WandbLogger
	from pytorch_lightning import Trainer

	# Add logging into your training_step (and elsewhere!).
	def training_step(self, batch, batch_idx):
		#...
		self.log('train/loss', loss)
		return loss

	# Add a WandbLogger to your Trainer.
	wandb_logger = WandbLogger()
	trainer = Trainer(logger=wandb_logger)

	# Fit your model.
	trainer.fit(model, mnist)
	"""

	import torch, torchvision
	import pytorch_lightning as pl

	class LitAutoEncoder(pl.LightningModule):
		def __init__(self, lr=1e-3, inp_size=28, optimizer="Adam"):
			super().__init__()

			self.encoder = torch.nn.Sequential(
				torch.nn.Linear(inp_size * inp_size, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
			)
			self.decoder = torch.nn.Sequential(
				torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, inp_size * inp_size)
			)
			self.lr = lr

			# Save hyperparameters to self.hparamsm auto-logged by wandb.
			self.save_hyperparameters()

		def training_step(self, batch, batch_idx):
			x, y = batch
			x = x.view(x.size(0), -1)
			z = self.encoder(x)
			x_hat = self.decoder(z)
			loss = torch.nn.functional.mse_loss(x_hat, x)

			# Log metrics to wandb.
			self.log("train_loss", loss)
			return loss

		def configure_optimizers(self):
			optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
			return optimizer

	# Init the autoencoder.
	autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

	# Setup data.
	batch_size = 32
	dataset = torchvision.datasets.MNIST(os.getcwd(), download=True, transform=torchvision.transforms.ToTensor())
	train_loader = torch.data.DataLoader(dataset, shuffle=True)

	# Initialise the wandb logger and name your wandb project.
	wandb_logger = pl.loggers.WandbLogger(project="my-awesome-project")

	# Add your batch size to the wandb config.
	wandb_logger.experiment.config["batch_size"] = batch_size

	# Pass wandb_logger to the Trainer.
	trainer = pl.Trainer(limit_train_batches=750, max_epochs=5, logger=wandb_logger)

	# Train the model.
	trainer.fit(model=autoencoder, train_dataloaders=train_loader)

	# [optional] finish the wandb run, necessary in notebooks.
	wandb.finish()

# REF [site] >> https://github.com/wandb/wandb
def keras_example():
	"""
	from wandb.keras import WandbCallback

	# Step1: Initialize W&B run.
	wandb.init(project="project_name")

	# 2. Save model inputs and hyperparameters.
	config = wandb.config
	config.learning_rate = 0.01

	# Model training code here ...

	# Step 3: Add WandbCallback.
	model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[WandbCallback()])
	"""

	import random
	import tensorflow as tf
	from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

	# Start a run, tracking hyperparameters.
	run = wandb.init(
		# Set the wandb project where this run will be logged.
		project="my-awesome-project",
		# Track hyperparameters and run metadata with wandb.config.
		config={
			"layer_1": 512,
			"activation_1": "relu",
			"dropout": random.uniform(0.01, 0.80),
			"layer_2": 10,
			"activation_2": "softmax",
			"optimizer": "sgd",
			"loss": "sparse_categorical_crossentropy",
			"metric": "accuracy",
			"epoch": 8,
			"batch_size": 256,
		},
	)

	# [optional] use wandb.config as your config.
	config = run.config

	# Get the data.
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	x_train, y_train = x_train[::5], y_train[::5]
	x_test, y_test = x_test[::20], y_test[::20]
	labels = [str(digit) for digit in range(np.max(y_train) + 1)]

	# Build a model.
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28, 28)),
		tf.keras.layers.Dense(config.layer_1, activation=config.activation_1),
		tf.keras.layers.Dropout(config.dropout),
		tf.keras.layers.Dense(config.layer_2, activation=config.activation_2),
	])

	# Compile the model.
	model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])

	# WandbMetricsLogger will log train and validation metrics to wandb.
	# WandbModelCheckpoint will upload model checkpoints to wandb.
	history = model.fit(
		x=x_train,
		y=y_train,
		epochs=config.epoch,
		batch_size=config.batch_size,
		validation_data=(x_test, y_test),
		callbacks=[
			WandbMetricsLogger(log_freq=5),
			WandbModelCheckpoint("models"),
		],
	)

	# [optional] finish the wandb run, necessary in notebooks.
	run.finish()

# REF [site] >> https://github.com/wandb/examples
def tensorflow_example():
	"""
	import tensorflow as tf

	# 1. Start a W&B run.
	wandb.init(project="gpt3")

	# 2. Save model inputs and hyperparameters.
	config = wandb.config
	config.learning_rate = 0.01

	# Model training here.

	# 3. Log metrics over time to visualize performance.
	with tf.Session() as sess:
		#...
		wandb.tensorflow.log(tf.summary.merge_all())
	"""

	raise NotImplementedError

# REF [site] >> https://github.com/wandb/wandb.
def huggingface_example():
	import datasets
	import transformers

	def tokenize_function(examples):
		return tokenizer(examples["text"], padding="max_length", truncation=True)

	def compute_metrics(eval_pred):
		logits, labels = eval_pred
		predictions = np.argmax(logits, axis=-1)
		return {"accuracy": np.mean(predictions == labels)}

	# Download prepare the data.
	dataset = datasets.load_dataset("yelp_review_full")
	tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")

	small_train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
	small_eval_dataset = dataset["test"].shuffle(seed=42).select(range(300))

	small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
	small_eval_dataset = small_train_dataset.map(tokenize_function, batched=True)

	# Download the model.
	model = transformers.AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

	# Set the wandb project where this run will be logged.
	os.environ["WANDB_PROJECT"] = "my-awesome-project"

	# Save your trained model checkpoint to wandb.
	os.environ["WANDB_LOG_MODEL"] = "true"

	# Turn off watch to log faster.
	os.environ["WANDB_WATCH"] = "false"

	# Pass "wandb" to the `report_to` parameter to turn on wandb logging.
	training_args = transformers.TrainingArguments(
		output_dir="models",
		report_to="wandb",
		logging_steps=5,
		per_device_train_batch_size=32,
		per_device_eval_batch_size=32,
		evaluation_strategy="steps",
		eval_steps=20,
		max_steps=100,
		save_steps=100,
	)

	# Define the trainer and start training.
	trainer = transformers.Trainer(
		model=model,
		args=training_args,
		train_dataset=small_train_dataset,
		eval_dataset=small_eval_dataset,
		compute_metrics=compute_metrics,
	)
	trainer.train()

	# [optional] finish the wandb run, necessary in notebooks.
	wandb.finish()

def main():
	# Log into W&B.
	#wandb.login()

	#quickstart()  # Not yet implemented.

	#simple_integration_with_framework_example()  # Not yet implemented.
	sklearn_example()
	#pytorch_example()  # Not yet implemented.
	#pytorch_lightning_example()
	#keras_example()
	#tensorflow_example()  # Not yet implemented.
	#huggingface_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
