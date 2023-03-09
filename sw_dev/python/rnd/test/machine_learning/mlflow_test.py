#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, random
import numpy as np
import mlflow

# REF [site] >> https://mlflow.org/docs/latest/quickstart.html
def quickstart():
	# Log a parameter (key-value pair).
	mlflow.log_param("param1", random.randint(0, 100))

	# Log a metric; metrics can be updated throughout the run.
	mlflow.log_metric("foo", random.random())
	mlflow.log_metric("foo", random.random() + 1)
	mlflow.log_metric("foo", random.random() + 2)

	# Log an artifact (output file).
	if not os.path.exists("./outputs"):
		os.makedirs("./outputs")
	with open("./outputs/test.txt", "w") as fd:
		fd.write("hello world!")
	mlflow.log_artifacts("./outputs")

# REF [site] >> https://github.com/mlflow/mlflow/tree/master/examples/sklearn_logistic_regression
def sklearn_logistic_regression_example():
	import mlflow.sklearn
	from sklearn.linear_model import LogisticRegression
	from urllib.parse import urlparse

	X = np.array([-2, -1, 0, 1, 2, 1]).reshape(-1, 1)
	y = np.array([0, 0, 1, 1, 1, 0])

	lr = LogisticRegression()
	lr.fit(X, y)

	score = lr.score(X, y)
	print(f"Score: {score}.")

	mlflow.log_metric("score", score)
	mlflow.sklearn.log_model(lr, "model")
	print(f"Model saved in run {mlflow.active_run().info.run_uuid}.")  # An MLflow run ID for that experiment.

# REF [site] >>
#	https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
#	https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_wine
def sklearn_elasticnet_wine_examaple():
	import sys, warnings, logging
	import pandas as pd
	from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import ElasticNet
	from urllib.parse import urlparse
	import mlflow.sklearn

	logging.basicConfig(level=logging.WARN)
	logger = logging.getLogger(__name__)

	def eval_metrics(actual, pred):
		rmse = np.sqrt(mean_squared_error(actual, pred))
		mae = mean_absolute_error(actual, pred)
		r2 = r2_score(actual, pred)
		return rmse, mae, r2

	warnings.filterwarnings("ignore")
	np.random.seed(40)

	# Read the wine-quality csv file from the URL.
	csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/data/winequality-red.csv"
	try:
		data = pd.read_csv(csv_url, sep=";")
	except Exception as ex:
		logger.exception(f"Unable to download training & test CSV, check your internet connection. Error: {ex}.")

	# Split the data into training and test sets. (0.75, 0.25) split.
	train, test = train_test_split(data)

	# The predicted column is "quality" which is a scalar from [3, 9].
	train_x = train.drop(["quality"], axis=1)
	test_x = test.drop(["quality"], axis=1)
	train_y = train[["quality"]]
	test_y = test[["quality"]]

	alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
	l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

	with mlflow.start_run():
		lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
		lr.fit(train_x, train_y)

		predicted_qualities = lr.predict(test_x)

		(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

		print(f"Elasticnet model (alpha={alpha:f}, l1_ratio={l1_ratio:f}):")
		print(f"  RMSE: {rmse}.")
		print(f"  MAE: {mae}.")
		print(f"  R2: {r2}.")

		mlflow.log_param("alpha", alpha)
		mlflow.log_param("l1_ratio", l1_ratio)
		mlflow.log_metric("rmse", rmse)
		mlflow.log_metric("r2", r2)
		mlflow.log_metric("mae", mae)

		# Model registry does not work with file store.
		tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
		if tracking_url_type_store != "file":
			# Register the model.
			# There are other ways to use the Model Registry, which depends on the use case,
			# please refer to the doc for more information: https://mlflow.org/docs/latest/model-registry.html#api-workflow.
			mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
		else:
			mlflow.sklearn.log_model(lr, "model")

# REF [site] >> https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
def pytorch_example():
	import torch
	import mlflow.pytorch

	class LinearNNModel(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.linear = torch.nn.Linear(1, 1)  # One in and one out.

		def forward(self, x):
			y_pred = self.linear(x)
			return y_pred

	def gen_data():
		# Example linear model modified to use y = 2x from https://github.com/hunkim/PyTorchZeroToAll.
		# X training data, y labels.
		X = torch.arange(1.0, 25.0).view(-1, 1)
		y = torch.from_numpy(np.array([x * 2 for x in X], dtype=np.float32)).view(-1, 1)
		return X, y

	# Define model, loss, and optimizer.
	model = LinearNNModel()
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

	# Training loop.
	epochs = 250
	X, y = gen_data()
	for epoch in range(epochs):
		# Forward pass: Compute predicted y by passing X to the model.
		y_pred = model(X)

		# Compute the loss.
		loss = criterion(y_pred, y)

		# Zero gradients, perform a backward pass, and update the weights.
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	#--------------------
	# Log the model.
	with mlflow.start_run() as run:
		mlflow.pytorch.log_model(model, "model")

		# Convert to scripted model and log the model.
		scripted_pytorch_model = torch.jit.script(model)
		mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

	#-----
	# Fetch the associated conda environment.
	env = mlflow.pytorch.get_default_conda_env()
	print(f"Conda env: {env}.")

	# Fetch the logged model artifacts.
	print(f"run_id: {run.info.run_id}.")
	for artifact_path in ["model/data", "scripted_model/data"]:
		artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, artifact_path)]
		print(f"Artifacts: {artifacts}.")

	#-----
	# Inference after loading the logged model.
	model_uri = "runs:/{}/model".format(run.info.run_id)
	loaded_model = mlflow.pytorch.load_model(model_uri)
	for x in [4.0, 6.0, 30.0]:
		X = torch.Tensor([[x]])
		y_pred = loaded_model(X)
		print(f"predict X: {x}, y_pred: {y_pred.data.item():.2f}.")

# REF [site] >> https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html
def pytorch_lightning_example():
	import torch, torchvision
	import pytorch_lightning as pl
	import torchmetrics
	import mlflow.pytorch

	class MNISTModel(pl.LightningModule):
		def __init__(self):
			super().__init__()
			self.l1 = torch.nn.Linear(28 * 28, 10)

		def forward(self, x):
			return torch.relu(self.l1(x.view(x.size(0), -1)))

		def training_step(self, batch, batch_nb):
			x, y = batch
			logits = self(x)
			loss = torch.nn.functional.cross_entropy(logits, y)
			pred = logits.argmax(dim=1)
			acc = torchmetrics.functional.accuracy(pred, y, task="multiclass", num_classes=10)

			# REF [site] >> https://mlflow.org/docs/latest/python_api/mlflow.client.html
			#self.logger.experiment.create_model_version(...)  # mlflow.MlflowClient.

			# Use the current of PyTorch logger.
			self.log("train_loss", loss, on_epoch=True)
			self.log("acc", acc, on_epoch=True)
			return loss

		def configure_optimizers(self):
			return torch.optim.Adam(self.parameters(), lr=0.02)

	def print_auto_logged_info(r):
		tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
		artifacts = [f.path for f in mlflow.MlflowClient().list_artifacts(r.info.run_id, "model")]
		print("run_id: {}".format(r.info.run_id))
		print("artifacts: {}".format(artifacts))
		print("params: {}".format(r.data.params))
		print("metrics: {}".format(r.data.metrics))
		print("tags: {}".format(tags))

	# Initialize our model.
	mnist_model = MNISTModel()

	# Initialize DataLoader from MNIST Dataset.
	train_ds = torchvision.datasets.MNIST(os.getcwd(), train=True, download=True, transform=torchvision.transforms.ToTensor())
	train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)

	# Initialize a trainer.
	if True:
		trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True)

		# Auto log all MLflow entities.
		mlflow.pytorch.autolog()
	else:
		mlflow_logger = pl.loggers.MLFlowLogger(experiment_name="lightning_logs", run_name=None, tracking_uri=None, tags=None, save_dir="./mlruns", log_model=False, prefix="", artifact_location=None, run_id=None)

		trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True, logger=mlflow_logger)

	# Train the model.
	with mlflow.start_run() as run:
		trainer.fit(mnist_model, train_loader)
		#trainer.fit(mnist_model, train_loader)

	# Fetch the auto logged parameters and metrics.
	print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

def main():
	# Logging functions:
	#	https://mlflow.org/docs/latest/tracking.html#logging-functions
	#
	#	mlflow.set_tracking_uri()
	#	mlflow.get_tracking_uri()
	#	mlflow.create_experiment()
	#	mlflow.set_experiment()
	#	mlflow.start_run()
	#	mlflow.end_run()
	#	mlflow.active_run()
	#	...

	try:
		# To log runs remotely, set the 'MLFLOW_TRACKING_URI' environment variable to a tracking server's URI or call mlflow.set_tracking_uri().
		#	Local file path (specified as file:/my/local/dir), where data is just directly stored locally.
		#	Database encoded as <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>. MLflow supports the dialects mysql, mssql, sqlite, and postgresql.
		#	HTTP server (specified as https://my-server:5000), which is a server hosting an MLflow tracking server.
		#	Databricks workspace (specified as databricks or as databricks://<profileName>, a Databricks CLI profile.
		# https://mlflow.org/docs/latest/tracking.html

		#mlflow.set_tracking_uri("./mlruns")  # The URI defaults to './mlruns'.
		#mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Run 'mlflow ui'.
		#mlflow.set_tracking_uri("sqlite:///mlruns.db")
		#tracking_uri = mlflow.get_tracking_uri()

		#exp_id = mlflow.create_experiment("my-experiment")
		#exp = mlflow.set_experiment("my-experiment")  # mlflow.entities.Experiment.

		#------------------------
		#quickstart()

		#sklearn_logistic_regression_example()
		#sklearn_elasticnet_wine_examaple()

		#pytorch_example()
		pytorch_lightning_example()
	except mlflow.exceptions.MlflowException as ex:
		print(f"mlflow.exceptions.MlflowException raised: {ex}.")
		raise

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
