#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#%matplotlib inline
import os, itertools, functools, json
from pathlib import Path
import numpy as np
import pandas as pd
import mxnet as mx
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.artificial import ComplexSeasonalTimeSeries
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.predictor import Predictor
from gluonts.mx import Trainer
from gluonts.mx import DistributionOutput, GaussianOutput
from gluonts.mx import MeanScaler, NOPScaler
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator
import matplotlib.pyplot as plt

# REF [site] >> https://ts.gluon.ai/tutorials/forecasting/quick_start_tutorial.html
def quick_start_tutorial():
	# Provided datasets.

	print(f"Available datasets: {list(dataset_recipes.keys())}")

	dataset = get_dataset("m4_hourly", regenerate=True)

	entry = next(iter(dataset.train))

	plt.figure()
	train_series = to_pandas(entry)
	train_series.plot()
	plt.grid(which="both")
	plt.legend(["train series"], loc="upper left")

	entry = next(iter(dataset.test))

	plt.figure()
	test_series = to_pandas(entry)
	test_series.plot()
	plt.axvline(train_series.index[-1], color="r")  # End of train dataset.
	plt.grid(which="both")
	plt.legend(["test series", "end of train series"], loc="upper left")

	plt.show()

	#--------------------
	# Custom datasets.

	N = 10  # Number of time series.
	T = 100  # Number of timesteps.
	prediction_length = 24
	freq = "1H"
	custom_dataset = np.random.normal(size=(N, T))
	start = pd.Timestamp("01-01-2019", freq=freq)  # Can be different for each time series.

	# Train dataset: cut the last window of length "prediction_length", add "target" and "start" fields.
	train_ds = ListDataset(
		[{"target": x, "start": start} for x in custom_dataset[:, :-prediction_length]],
		freq=freq
	)
	# Test dataset: use the whole dataset, add "target" and "start" fields.
	test_ds = ListDataset(
		[{"target": x, "start": start} for x in custom_dataset],
		freq=freq
	)

	#--------------------
	# Training an existing model (Estimator).

	estimator = SimpleFeedForwardEstimator(
		num_hidden_dimensions=[10],
		prediction_length=dataset.metadata.prediction_length,
		context_length=100,
		freq=dataset.metadata.freq,
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			num_batches_per_epoch=100
		)
	)

	predictor = estimator.train(dataset.train)

	#--------------------
	# Visualize and evaluate forecasts.

	forecast_it, ts_it = make_evaluation_predictions(
		dataset=dataset.test,  # Test dataset.
		predictor=predictor,  # Predictor.
		num_samples=100,  # Number of sample paths we want for evaluation.
	)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	# First entry of the time series list.
	ts_entry = tss[0]

	# First 5 values of the time series (convert from pandas to numpy).
	print(np.array(ts_entry[:5]).reshape(-1,))

	# First entry of dataset.test.
	dataset_test_entry = next(iter(dataset.test))

	# First 5 values.
	print(dataset_test_entry["target"][:5])

	# First entry of the forecast list.
	forecast_entry = forecasts[0]

	print(f"Number of sample paths: {forecast_entry.num_samples}")
	print(f"Dimension of samples: {forecast_entry.samples.shape}")
	print(f"Start date of the forecast window: {forecast_entry.start_date}")
	print(f"Frequency of the time series: {forecast_entry.freq}")

	print(f"Mean of the future window:\n {forecast_entry.mean}")
	print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

	def plot_prob_forecasts(ts_entry, forecast_entry):
		plot_length = 150
		prediction_intervals = (50.0, 90.0)
		legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

		fig, ax = plt.subplots(1, 1, figsize=(10, 7))
		ts_entry[-plot_length:].plot(ax=ax)  # Plot the time series.
		forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
		plt.grid(which="both")
		plt.legend(legend, loc="upper left")
		plt.show()

	plot_prob_forecasts(ts_entry, forecast_entry)

	evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
	agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))

	print(json.dumps(agg_metrics, indent=4))
	print(item_metrics.head())

	item_metrics.plot(x="MSIS", y="MASE", kind="scatter")
	plt.grid(which="both")

	plt.show()

# REF [site] >> https://ts.gluon.ai/tutorials/forecasting/extended_tutorial.html
def extended_forecasting_tutorial():
	mx.random.seed(0)
	np.random.seed(0)

	print(f"Available datasets: {list(dataset_recipes.keys())}")

	dataset = get_dataset("m4_hourly", regenerate=True)

	# Get the first time series in the training set.
	train_entry = next(iter(dataset.train))
	print(train_entry.keys())

	# Get the first time series in the test set.
	test_entry = next(iter(dataset.test))
	print(test_entry.keys())

	test_series = to_pandas(test_entry)
	train_series = to_pandas(train_entry)

	fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

	train_series.plot(ax=ax[0])
	ax[0].grid(which="both")
	ax[0].legend(["train series"], loc="upper left")

	test_series.plot(ax=ax[1])
	ax[1].axvline(train_series.index[-1], color="r")  # End of train dataset.
	ax[1].grid(which="both")
	ax[1].legend(["test series", "end of train series"], loc="upper left")

	plt.show()

	print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")
	print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
	print(f"Frequency of the time series: {dataset.metadata.freq}")

	#--------------------
	# Create artificial datasets.

	artificial_dataset = ComplexSeasonalTimeSeries(
		num_series=10,
		prediction_length=21,
		freq_str="H",
		length_low=30,
		length_high=200,
		min_val=-10000,
		max_val=10000,
		is_integer=False,
		proportion_missing_values=0,
		is_noise=True,
		is_scale=True,
		percentage_unique_timestamps=1,
		is_out_of_bounds_date=True,
	)

	print(f"prediction length: {artificial_dataset.metadata.prediction_length}")
	print(f"frequency: {artificial_dataset.metadata.freq}")

	print(f"type of train dataset: {type(artificial_dataset.train)}")
	print(f"train dataset fields: {artificial_dataset.train[0].keys()}")
	print(f"type of test dataset: {type(artificial_dataset.test)}")
	print(f"test dataset fields: {artificial_dataset.test[0].keys()}")

	train_ds = ListDataset(
		artificial_dataset.train,
		freq=artificial_dataset.metadata.freq
	)

	test_ds = ListDataset(
		artificial_dataset.test,
		freq=artificial_dataset.metadata.freq
	)

	train_entry = next(iter(train_ds))
	print(train_entry.keys())

	test_entry = next(iter(test_ds))
	print(test_entry.keys())

	test_series = to_pandas(test_entry)
	train_series = to_pandas(train_entry)

	fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

	train_series.plot(ax=ax[0])
	ax[0].grid(which="both")
	ax[0].legend(["train series"], loc="upper left")

	test_series.plot(ax=ax[1])
	ax[1].axvline(train_series.index[-1], color="r")  # End of train dataset.
	ax[1].grid(which="both")
	ax[1].legend(["test series", "end of train series"], loc="upper left")

	plt.show()

	#--------------------
	# Use your time series and features.

	[f"FieldName.{k} = '{v}'" for k, v in FieldName.__dict__.items() if not k.startswith("_")]

	def create_dataset(num_series, num_steps, period=24, mu=1, sigma=0.3):
		# Create target: noise + pattern.
		# Noise.
		noise = np.random.normal(mu, sigma, size=(num_series, num_steps))

		# Pattern - sinusoid with different phase.
		sin_minusPi_Pi = np.sin(np.tile(np.linspace(-np.pi, np.pi, period), int(num_steps / period)))
		sin_Zero_2Pi = np.sin(np.tile(np.linspace(0, 2 * np.pi, 24), int(num_steps / period)))

		pattern = np.concatenate(
			(
				np.tile(
					sin_minusPi_Pi.reshape(1, -1),
					(int(np.ceil(num_series / 2)),1)
				),
				np.tile(
					sin_Zero_2Pi.reshape(1, -1),
					(int(np.floor(num_series / 2)), 1)
				)
			),
			axis=0
		)

		target = noise + pattern

		# Create time features: use target one period earlier, append with zeros.
		feat_dynamic_real = np.concatenate(
			(
				np.zeros((num_series, period)),
				target[:, :-period]
			),
			axis=1
		)

		# Create categorical static feats: use the sinusoid type as a categorical feature.
		feat_static_cat = np.concatenate(
			(
				np.zeros(int(np.ceil(num_series / 2))),
				np.ones(int(np.floor(num_series / 2)))
			),
			axis=0
		)

		return target, feat_dynamic_real, feat_static_cat

	# Define the parameters of the dataset.
	custom_ds_metadata = {
		"num_series": 100,
		"num_steps": 24 * 7,
		"prediction_length": 24,
		"freq": "1H",
		"start": [
			pd.Timestamp("01-01-2019", freq="1H")
			for _ in range(100)
		]
	}

	data_out = create_dataset(
		custom_ds_metadata["num_series"],
		custom_ds_metadata["num_steps"],
		custom_ds_metadata["prediction_length"]
	)

	target, feat_dynamic_real, feat_static_cat = data_out

	train_ds = ListDataset(
		[
			{
				FieldName.TARGET: target,
				FieldName.START: start,
				FieldName.FEAT_DYNAMIC_REAL: [fdr],
				FieldName.FEAT_STATIC_CAT: [fsc]
			}
			for (target, start, fdr, fsc) in zip(
				target[:, :-custom_ds_metadata["prediction_length"]],
				custom_ds_metadata["start"],
				feat_dynamic_real[:, :-custom_ds_metadata["prediction_length"]],
				feat_static_cat
			)
		],
		freq=custom_ds_metadata["freq"]
	)

	test_ds = ListDataset(
		[
			{
				FieldName.TARGET: target,
				FieldName.START: start,
				FieldName.FEAT_DYNAMIC_REAL: [fdr],
				FieldName.FEAT_STATIC_CAT: [fsc]
			}
			for (target, start, fdr, fsc) in zip(
				target,
				custom_ds_metadata["start"],
				feat_dynamic_real,
				feat_static_cat)
		],
		freq=custom_ds_metadata["freq"]
	)

	train_entry = next(iter(train_ds))
	print(train_entry.keys())

	test_entry = next(iter(test_ds))
	print(test_entry.keys())

	test_series = to_pandas(test_entry)
	train_series = to_pandas(train_entry)

	fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

	train_series.plot(ax=ax[0])
	ax[0].grid(which="both")
	ax[0].legend(["train series"], loc="upper left")

	test_series.plot(ax=ax[1])
	ax[1].axvline(train_series.index[-1], color="r")  # End of train dataset
	ax[1].grid(which="both")
	ax[1].legend(["test series", "end of train series"], loc="upper left")

	plt.show()

	#--------------------
	# Transformations.

	from gluonts.transform import (
		AddAgeFeature,
		AddObservedValuesIndicator,
		Chain,
		ExpectedNumInstanceSampler,
		InstanceSplitter,
		SetFieldIfNotPresent,
	)

	# Define a transformation.
	def create_transformation(freq, context_length, prediction_length):
		return Chain(
			[
				AddObservedValuesIndicator(
					target_field=FieldName.TARGET,
					output_field=FieldName.OBSERVED_VALUES,
				),
				AddAgeFeature(
					target_field=FieldName.TARGET,
					output_field=FieldName.FEAT_AGE,
					pred_length=prediction_length,
					log_scale=True,
				),
				InstanceSplitter(
					target_field=FieldName.TARGET,
					is_pad_field=FieldName.IS_PAD,
					start_field=FieldName.START,
					forecast_start_field=FieldName.FORECAST_START,
					instance_sampler=ExpectedNumInstanceSampler(
						num_instances=1,
						min_future=prediction_length,
					),
					past_length=context_length,
					future_length=prediction_length,
					time_series_fields=[
						FieldName.FEAT_AGE,
						FieldName.FEAT_DYNAMIC_REAL,
						FieldName.OBSERVED_VALUES,
					],
				),
			]
		)

	# Transform a dataset.
	transformation = create_transformation(
		custom_ds_metadata["freq"],
		2 * custom_ds_metadata["prediction_length"],  # Can be any appropriate value.
		custom_ds_metadata["prediction_length"]
	)

	train_tf = transformation(iter(train_ds), is_train=True)
	type(train_tf)

	train_tf_entry = next(iter(train_tf))
	print([k for k in train_tf_entry.keys()])

	print(f"past target shape: {train_tf_entry['past_target'].shape}")
	print(f"future target shape: {train_tf_entry['future_target'].shape}")
	print(f"past observed values shape: {train_tf_entry['past_observed_values'].shape}")
	print(f"future observed values shape: {train_tf_entry['future_observed_values'].shape}")
	print(f"past age feature shape: {train_tf_entry['past_feat_dynamic_age'].shape}")
	print(f"future age feature shape: {train_tf_entry['future_feat_dynamic_age'].shape}")
	print(train_tf_entry["feat_static_cat"])

	print([k for k in next(iter(train_ds)).keys()])

	test_tf = transformation(iter(test_ds), is_train=False)

	test_tf_entry = next(iter(test_tf))
	print([k for k in test_tf_entry.keys()])

	print(f"past target shape: {test_tf_entry['past_target'].shape}")
	print(f"future target shape: {test_tf_entry['future_target'].shape}")
	print(f"past observed values shape: {test_tf_entry['past_observed_values'].shape}")
	print(f"future observed values shape: {test_tf_entry['future_observed_values'].shape}")
	print(f"past age feature shape: {test_tf_entry['past_feat_dynamic_age'].shape}")
	print(f"future age feature shape: {test_tf_entry['future_feat_dynamic_age'].shape}")
	print(test_tf_entry["feat_static_cat"])

	#--------------------
	# Training an existing model.

	# Configuring an estimator.
	estimator = SimpleFeedForwardEstimator(
		num_hidden_dimensions=[10],
		prediction_length=custom_ds_metadata["prediction_length"],
		context_length=2*custom_ds_metadata["prediction_length"],
		freq=custom_ds_metadata["freq"],
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			hybridize=False,
			num_batches_per_epoch=100
		)
	)

	# Getting a predictor.
	predictor = estimator.train(train_ds)

	#--------------------
	# Saving/Loading an existing model.

	# Save the trained model in tmp/.
	predictor.serialize(Path("/tmp/"))

	# Loads it back.
	predictor_deserialized = Predictor.deserialize(Path("/tmp/"))

	#--------------------
	# Evaluation.

	# Getting the forecasts.
	forecast_it, ts_it = make_evaluation_predictions(
		dataset=test_ds,  # Test dataset.
		predictor=predictor,  # Predictor.
		num_samples=100,  # Number of sample paths we want for evaluation.
	)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	# First entry of the time series list
	ts_entry = tss[0]

	# First 5 values of the time series (convert from pandas to numpy)
	np.array(ts_entry[:5]).reshape(-1,)

	# First entry of test_ds
	test_ds_entry = next(iter(test_ds))

	# First 5 values
	test_ds_entry["target"][:5]

	# First entry of the forecast list
	forecast_entry = forecasts[0]

	print(f"Number of sample paths: {forecast_entry.num_samples}")
	print(f"Dimension of samples: {forecast_entry.samples.shape}")
	print(f"Start date of the forecast window: {forecast_entry.start_date}")
	print(f"Frequency of the time series: {forecast_entry.freq}")

	print(f"Mean of the future window:\n {forecast_entry.mean}")
	print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

	def plot_prob_forecasts(ts_entry, forecast_entry):
		plot_length = 150
		prediction_intervals = (50.0, 90.0)
		legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

		fig, ax = plt.subplots(1, 1, figsize=(10, 7))
		ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
		forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
		plt.grid(which="both")
		plt.legend(legend, loc="upper left")
		plt.show()

	plot_prob_forecasts(ts_entry, forecast_entry)

	# Compute metrics.
	evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
	agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))

	print(json.dumps(agg_metrics, indent=4))

	print(item_metrics.head())

	item_metrics.plot(x="MSIS", y="MASE", kind="scatter")
	plt.grid(which="both")
	plt.show()

	#--------------------
	# Create your own model.

	from gluonts.core.component import validated
	from gluonts.dataset.loader import TrainDataLoader
	from gluonts.mx import (
		as_in_context,
		batchify,
		copy_parameters,
		get_hybrid_forward_input_names,
		GluonEstimator,
		RepresentableBlockPredictor,
	)
	from gluonts.transform import (
		ExpectedNumInstanceSampler,
		Transformation,
		InstanceSplitter,
		TestSplitSampler,
		SelectFields,
		Chain
	)

	# Point forecasts with a simple feedforward network.
	class MyNetwork(mx.gluon.HybridBlock):
		def __init__(self, prediction_length, num_cells, **kwargs):
			super().__init__(**kwargs)
			self.prediction_length = prediction_length
			self.num_cells = num_cells

			with self.name_scope():
				# Set up a 3 layer neural network that directly predicts the target values.
				self.nn = mx.gluon.nn.HybridSequential()
				self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation="relu"))
				self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation="relu"))
				self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length, activation="softrelu"))

	class MyTrainNetwork(MyNetwork):
		def hybrid_forward(self, F, past_target, future_target):
			prediction = self.nn(past_target)
			# Calculate L1 loss with the future_target to learn the median.
			return (prediction - future_target).abs().mean(axis=-1)

	class MyPredNetwork(MyTrainNetwork):
		# The prediction network only receives past_target and returns predictions.
		def hybrid_forward(self, F, past_target):
			prediction = self.nn(past_target)
			return prediction.expand_dims(axis=1)

	class MyEstimator(GluonEstimator):
		@validated()
		def __init__(
			self,
			prediction_length: int,
			context_length: int,
			freq: str,
			num_cells: int,
			batch_size: int = 32,
			trainer: Trainer = Trainer()
		) -> None:
			super().__init__(trainer=trainer, batch_size=batch_size)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.freq = freq
			self.num_cells = num_cells

		def create_transformation(self):
			return Chain([])

		def create_training_data_loader(self, dataset, **kwargs):
			instance_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=ExpectedNumInstanceSampler(
					num_instances=1,
					min_future=self.prediction_length
				),
				past_length=self.context_length,
				future_length=self.prediction_length,
			)
			input_names = get_hybrid_forward_input_names(MyTrainNetwork)
			return TrainDataLoader(
				dataset=dataset,
				transform=instance_splitter + SelectFields(input_names),
				batch_size=self.batch_size,
				stack_fn=functools.partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
				decode_fn=functools.partial(as_in_context, ctx=self.trainer.ctx),
				**kwargs,
			)

		def create_training_network(self) -> MyTrainNetwork:
			return MyTrainNetwork(
				prediction_length=self.prediction_length,
				num_cells = self.num_cells
			)

		def create_predictor(
			self, transformation: Transformation, trained_network: mx.gluon.HybridBlock
		) -> Predictor:
			prediction_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=TestSplitSampler(),
				past_length=self.context_length,
				future_length=self.prediction_length,
			)

			prediction_network = MyPredNetwork(
				prediction_length=self.prediction_length,
				num_cells=self.num_cells
			)

			copy_parameters(trained_network, prediction_network)

			return RepresentableBlockPredictor(
				input_transform=transformation + prediction_splitter,
				prediction_net=prediction_network,
				batch_size=self.trainer.batch_size,
				freq=self.freq,
				prediction_length=self.prediction_length,
				ctx=self.trainer.ctx,
			)

	estimator = MyEstimator(
		prediction_length=custom_ds_metadata["prediction_length"],
		context_length=2*custom_ds_metadata["prediction_length"],
		freq=custom_ds_metadata["freq"],
		num_cells=40,
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			hybridize=False,
			num_batches_per_epoch=100
		)
	)

	predictor = estimator.train(train_ds)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	plot_prob_forecasts(tss[0], forecasts[0])

	# Probabilistic forecasting.
	class MyProbNetwork(mx.gluon.HybridBlock):
		def __init__(
			self,
			prediction_length,
			distr_output,
			num_cells,
			num_sample_paths=100,
			**kwargs
		) -> None:
			super().__init__(**kwargs)
			self.prediction_length = prediction_length
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_sample_paths = num_sample_paths
			self.proj_distr_args = distr_output.get_args_proj()

			with self.name_scope():
				# Set up a 2 layer neural network that its ouput will be projected to the distribution parameters.
				self.nn = mx.gluon.nn.HybridSequential()
				self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation="relu"))
				self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length * self.num_cells, activation="relu"))

	class MyProbTrainNetwork(MyProbNetwork):
		def hybrid_forward(self, F, past_target, future_target):
			# Compute network output.
			net_output = self.nn(past_target)

			# (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features).
			net_output = net_output.reshape(0, self.prediction_length, -1)

			# Project network output to distribution parameters domain.
			distr_args = self.proj_distr_args(net_output)

			# Compute distribution.
			distr = self.distr_output.distribution(distr_args)

			# Negative log-likelihood.
			loss = distr.loss(future_target)
			return loss

	class MyProbPredNetwork(MyProbTrainNetwork):
		# The prediction network only receives past_target and returns predictions.
		def hybrid_forward(self, F, past_target):
			# Repeat past target: from (batch_size, past_target_length) to
			# (batch_size * num_sample_paths, past_target_length).
			repeated_past_target = past_target.repeat(
				repeats=self.num_sample_paths, axis=0
			)

			# Compute network output.
			net_output = self.nn(repeated_past_target)

			# (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features).
			net_output = net_output.reshape(0, self.prediction_length, -1)

			# Rroject network output to distribution parameters domain.
			distr_args = self.proj_distr_args(net_output)

			# Compute distribution.
			distr = self.distr_output.distribution(distr_args)

			# Get (batch_size * num_sample_paths, prediction_length) samples.
			samples = distr.sample()

			# Reshape from (batch_size * num_sample_paths, prediction_length) to
			# (batch_size, num_sample_paths, prediction_length).
			return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))

	class MyProbEstimator(GluonEstimator):
		@validated()
		def __init__(
			self,
			prediction_length: int,
			context_length: int,
			freq: str,
			distr_output: DistributionOutput,
			num_cells: int,
			num_sample_paths: int = 100,
			batch_size: int = 32,
			trainer: Trainer = Trainer()
		) -> None:
			super().__init__(trainer=trainer, batch_size=batch_size)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.freq = freq
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_sample_paths = num_sample_paths

		def create_transformation(self):
			return Chain([])

		def create_training_data_loader(self, dataset, **kwargs):
			instance_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=ExpectedNumInstanceSampler(
					num_instances=1,
					min_future=self.prediction_length
				),
				past_length=self.context_length,
				future_length=self.prediction_length,
			)
			input_names = get_hybrid_forward_input_names(MyProbTrainNetwork)
			return TrainDataLoader(
				dataset=dataset,
				transform=instance_splitter + SelectFields(input_names),
				batch_size=self.batch_size,
				stack_fn=functools.partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
				decode_fn=functools.partial(as_in_context, ctx=self.trainer.ctx),
				**kwargs,
			)

		def create_training_network(self) -> MyProbTrainNetwork:
			return MyProbTrainNetwork(
				prediction_length=self.prediction_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_sample_paths=self.num_sample_paths
			)

		def create_predictor(
			self, transformation: Transformation, trained_network: mx.gluon.HybridBlock
		) -> Predictor:
			prediction_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=TestSplitSampler(),
				past_length=self.context_length,
				future_length=self.prediction_length,
			)

			prediction_network = MyProbPredNetwork(
				prediction_length=self.prediction_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_sample_paths=self.num_sample_paths
			)

			copy_parameters(trained_network, prediction_network)

			return RepresentableBlockPredictor(
				input_transform=transformation + prediction_splitter,
				prediction_net=prediction_network,
				batch_size=self.trainer.batch_size,
				freq=self.freq,
				prediction_length=self.prediction_length,
				ctx=self.trainer.ctx,
			)

	estimator = MyProbEstimator(
		prediction_length=custom_ds_metadata["prediction_length"],
		context_length=2*custom_ds_metadata["prediction_length"],
		freq=custom_ds_metadata["freq"],
		distr_output=GaussianOutput(),
		num_cells=40,
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			hybridize=False,
			num_batches_per_epoch=100
		)
	)

	predictor = estimator.train(train_ds)

	forecast_it, ts_it = make_evaluation_predictions(
		dataset=test_ds,  # Test dataset.
		predictor=predictor,  # Predictor.
		num_samples=100,  # Number of sample paths we want for evaluation.
	)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	plot_prob_forecasts(tss[0], forecasts[0])

	#--------------------
	# Add features and scaling.

	class MyProbNetwork(mx.gluon.HybridBlock):
		def __init__(
			self,
			prediction_length,
			context_length,
			distr_output,
			num_cells,
			num_sample_paths=100,
			scaling=True,
			**kwargs
		) -> None:
			super().__init__(**kwargs)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_sample_paths = num_sample_paths
			self.proj_distr_args = distr_output.get_args_proj()
			self.scaling = scaling

			with self.name_scope():
				# Set up a 2 layer neural network that its ouput will be projected to the distribution parameters.
				self.nn = mx.gluon.nn.HybridSequential()
				self.nn.add(mx.gluon.nn.Dense(units=self.num_cells, activation="relu"))
				self.nn.add(mx.gluon.nn.Dense(units=self.prediction_length * self.num_cells, activation="relu"))

				if scaling:
					self.scaler = MeanScaler(keepdims=True)
				else:
					self.scaler = NOPScaler(keepdims=True)

		def compute_scale(self, past_target, past_observed_values):
			# Scale shape is (batch_size, 1).
			_, scale = self.scaler(
				past_target.slice_axis(
					axis=1, begin=-self.context_length, end=None
				),
				past_observed_values.slice_axis(
					axis=1, begin=-self.context_length, end=None
				),
			)

			return scale

	class MyProbTrainNetwork(MyProbNetwork):
		def hybrid_forward(self, F, past_target, future_target, past_observed_values, past_feat_dynamic_real):
			# Compute scale.
			scale = self.compute_scale(past_target, past_observed_values)

			# Scale target and time features.
			past_target_scale = F.broadcast_div(past_target, scale)
			past_feat_dynamic_real_scale = F.broadcast_div(past_feat_dynamic_real.squeeze(axis=-1), scale)

			# Concatenate target and time features to use them as input to the network.
			net_input = F.concat(past_target_scale, past_feat_dynamic_real_scale, dim=-1)

			# Compute network output.
			net_output = self.nn(net_input)

			# (batch, prediction_length * nn_features)  ->  (batch, prediction_length, nn_features).
			net_output = net_output.reshape(0, self.prediction_length, -1)

			# Project network output to distribution parameters domain.
			distr_args = self.proj_distr_args(net_output)

			# Compute distribution.
			distr = self.distr_output.distribution(distr_args, scale=scale)

			# Negative log-likelihood.
			loss = distr.loss(future_target)
			return loss

	class MyProbPredNetwork(MyProbTrainNetwork):
		# The prediction network only receives past_target and returns predictions.
		def hybrid_forward(self, F, past_target, past_observed_values, past_feat_dynamic_real):
			# Repeat fields: from (batch_size, past_target_length) to
			# (batch_size * num_sample_paths, past_target_length).
			repeated_past_target = past_target.repeat(
				repeats=self.num_sample_paths, axis=0
			)
			repeated_past_observed_values = past_observed_values.repeat(
				repeats=self.num_sample_paths, axis=0
			)
			repeated_past_feat_dynamic_real = past_feat_dynamic_real.repeat(
				repeats=self.num_sample_paths, axis=0
			)

			# Compute scale.
			scale = self.compute_scale(repeated_past_target, repeated_past_observed_values)

			# Scale repeated target and time features.
			repeated_past_target_scale = F.broadcast_div(repeated_past_target, scale)
			repeated_past_feat_dynamic_real_scale = F.broadcast_div(repeated_past_feat_dynamic_real.squeeze(axis=-1), scale)

			# Concatenate target and time features to use them as input to the network.
			net_input = F.concat(repeated_past_target_scale, repeated_past_feat_dynamic_real_scale, dim=-1)

			# Compute network oputput.
			net_output = self.nn(net_input)

			# (batch * num_sample_paths, prediction_length * nn_features)  ->  (batch * num_sample_paths, prediction_length, nn_features).
			net_output = net_output.reshape(0, self.prediction_length, -1)

			# Project network output to distribution parameters domain.
			distr_args = self.proj_distr_args(net_output)

			# Pompute distribution.
			distr = self.distr_output.distribution(distr_args, scale=scale)

			# Get (batch_size * num_sample_paths, prediction_length) samples.
			samples = distr.sample()

			# Reshape from (batch_size * num_sample_paths, prediction_length) to
			# (batch_size, num_sample_paths, prediction_length).
			return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))

	class MyProbEstimator(GluonEstimator):
		@validated()
		def __init__(
			self,
			prediction_length: int,
			context_length: int,
			freq: str,
			distr_output: DistributionOutput,
			num_cells: int,
			num_sample_paths: int = 100,
			scaling: bool = True,
			batch_size: int = 32,
			trainer: Trainer = Trainer()
		) -> None:
			super().__init__(trainer=trainer, batch_size=batch_size)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.freq = freq
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_sample_paths = num_sample_paths
			self.scaling = scaling

		def create_transformation(self):
			# Feature transformation that the model uses for input.
			return AddObservedValuesIndicator(
				target_field=FieldName.TARGET,
				output_field=FieldName.OBSERVED_VALUES,
			)

		def create_training_data_loader(self, dataset, **kwargs):
			instance_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=ExpectedNumInstanceSampler(
					num_instances=1,
					min_future=self.prediction_length
				),
				past_length=self.context_length,
				future_length=self.prediction_length,
				time_series_fields=[
					FieldName.FEAT_DYNAMIC_REAL,
					FieldName.OBSERVED_VALUES,
				],
			)
			input_names = get_hybrid_forward_input_names(MyProbTrainNetwork)
			return TrainDataLoader(
				dataset=dataset,
				transform=instance_splitter + SelectFields(input_names),
				batch_size=self.batch_size,
				stack_fn=functools.partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
				decode_fn=functools.partial(as_in_context, ctx=self.trainer.ctx),
				**kwargs,
			)

		def create_training_network(self) -> MyProbTrainNetwork:
			return MyProbTrainNetwork(
				prediction_length=self.prediction_length,
				context_length=self.context_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_sample_paths=self.num_sample_paths,
				scaling=self.scaling
			)

		def create_predictor(
			self, transformation: Transformation, trained_network: mx.gluon.HybridBlock
		) -> Predictor:
			prediction_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=TestSplitSampler(),
				past_length=self.context_length,
				future_length=self.prediction_length,
				time_series_fields=[
					FieldName.FEAT_DYNAMIC_REAL,
					FieldName.OBSERVED_VALUES,
				],
			)

			prediction_network = MyProbPredNetwork(
				prediction_length=self.prediction_length,
				context_length=self.context_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_sample_paths=self.num_sample_paths,
				scaling=self.scaling
			)

			copy_parameters(trained_network, prediction_network)

			return RepresentableBlockPredictor(
				input_transform=transformation + prediction_splitter,
				prediction_net=prediction_network,
				batch_size=self.trainer.batch_size,
				freq=self.freq,
				prediction_length=self.prediction_length,
				ctx=self.trainer.ctx,
			)

	estimator = MyProbEstimator(
		prediction_length=custom_ds_metadata["prediction_length"],
		context_length=2*custom_ds_metadata["prediction_length"],
		freq=custom_ds_metadata["freq"],
		distr_output=GaussianOutput(),
		num_cells=40,
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			hybridize=False,
			num_batches_per_epoch=100
		)
	)

	predictor = estimator.train(train_ds)

	forecast_it, ts_it = make_evaluation_predictions(
		dataset=test_ds,  # Test dataset.
		predictor=predictor,  # Predictor.
		num_samples=100,  # Number of sample paths we want for evaluation.
	)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	plot_prob_forecasts(tss[0], forecasts[0])

	#--------------------
	# From feedforward to RNN.

	class MyProbRNN(mx.gluon.HybridBlock):
		def __init__(self,
			prediction_length,
			context_length,
			distr_output,
			num_cells,
			num_layers,
			num_sample_paths=100,
			scaling=True,
			**kwargs
		) -> None:
			super().__init__(**kwargs)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_layers = num_layers
			self.num_sample_paths = num_sample_paths
			self.proj_distr_args = distr_output.get_args_proj()
			self.scaling = scaling

			with self.name_scope():
				self.rnn = mx.gluon.rnn.HybridSequentialRNNCell()
				for k in range(self.num_layers):
					cell = mx.gluon.rnn.LSTMCell(hidden_size=self.num_cells)
					cell = mx.gluon.rnn.ResidualCell(cell) if k > 0 else cell
					self.rnn.add(cell)

				if scaling:
					self.scaler = MeanScaler(keepdims=True)
				else:
					self.scaler = NOPScaler(keepdims=True)

		def compute_scale(self, past_target, past_observed_values):
			# Scale is computed on the context length last units of the past target
			# scale shape is (batch_size, 1, *target_shape).
			_, scale = self.scaler(
				past_target.slice_axis(
					axis=1, begin=-self.context_length, end=None
				),
				past_observed_values.slice_axis(
					axis=1, begin=-self.context_length, end=None
				),
			)

			return scale

		def unroll_encoder(
			self,
			F,
			past_target,
			past_observed_values,
			future_target=None,
			future_observed_values=None
		):
			# Overall target field.
			# Input target from -(context_length + prediction_length + 1) to -1.
			if future_target is not None:  # during training
				target_in = F.concat(
					past_target, future_target, dim=-1
				).slice_axis(
					axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
				)

				# Overall observed_values field.
				# Input observed_values corresponding to target_in.
				observed_values_in = F.concat(
					past_observed_values, future_observed_values, dim=-1
				).slice_axis(
					axis=1, begin=-(self.context_length + self.prediction_length + 1), end=-1
				)

				rnn_length = self.context_length + self.prediction_length
			else:  # During inference.
				target_in = past_target.slice_axis(
					axis=1, begin=-(self.context_length + 1), end=-1
				)

				# Overall observed_values field.
				# Input observed_values corresponding to target_in.
				observed_values_in = past_observed_values.slice_axis(
					axis=1, begin=-(self.context_length + 1), end=-1
				)

				rnn_length = self.context_length

			# Compute scale.
			scale = self.compute_scale(target_in, observed_values_in)

			# Scale target_in.
			target_in_scale = F.broadcast_div(target_in, scale)

			# Compute network output.
			net_output, states = self.rnn.unroll(
				inputs=target_in_scale,
				length=rnn_length,
				layout="NTC",
				merge_outputs=True,
			)

			return net_output, states, scale

	class MyProbTrainRNN(MyProbRNN):
		def hybrid_forward(
			self,
			F,
			past_target,
			future_target,
			past_observed_values,
			future_observed_values
		):
			net_output, _, scale = self.unroll_encoder(
				F, past_target, past_observed_values, future_target, future_observed_values
			)

			# Output target from -(context_length + prediction_length) to end.
			target_out = F.concat(
				past_target, future_target, dim=-1
			).slice_axis(
				axis=1, begin=-(self.context_length + self.prediction_length), end=None
			)

			# Project network output to distribution parameters domain.
			distr_args = self.proj_distr_args(net_output)

			# Compute distribution
			distr = self.distr_output.distribution(distr_args, scale=scale)

			# Negative log-likelihood.
			loss = distr.loss(target_out)
			return loss

	class MyProbPredRNN(MyProbTrainRNN):
		def sample_decoder(self, F, past_target, states, scale):
			# Repeat fields: from (batch_size, past_target_length) to
			# (batch_size * num_sample_paths, past_target_length).
			repeated_states = [
				s.repeat(repeats=self.num_sample_paths, axis=0)
				for s in states
			]
			repeated_scale = scale.repeat(repeats=self.num_sample_paths, axis=0)

			# First decoder input is the last value of the past_target, i.e.,
			# the previous value of the first time step we want to forecast.
			decoder_input = past_target.slice_axis(
				axis=1, begin=-1, end=None
			).repeat(
				repeats=self.num_sample_paths, axis=0
			)

			# List with samples at each time step.
			future_samples = []

			# For each future time step we draw new samples for this time step and update the state
			# the drawn samples are the inputs to the rnn at the next time step.
			for k in range(self.prediction_length):
				rnn_outputs, repeated_states = self.rnn.unroll(
					inputs=decoder_input,
					length=1,
					begin_state=repeated_states,
					layout="NTC",
					merge_outputs=True,
				)

				# Project network output to distribution parameters domain.
				distr_args = self.proj_distr_args(rnn_outputs)

				# Compute distribution.
				distr = self.distr_output.distribution(distr_args, scale=repeated_scale)

				# Draw samples (batch_size * num_samples, 1).
				new_samples = distr.sample()

				# Append the samples of the current time step.
				future_samples.append(new_samples)

				# Update decoder input for the next time step.
				decoder_input = new_samples

			samples = F.concat(*future_samples, dim=1)

			# (batch_size, num_samples, prediction_length).
			return samples.reshape(shape=(-1, self.num_sample_paths, self.prediction_length))

		def hybrid_forward(self, F, past_target, past_observed_values):
			# Unroll encoder over context_length.
			net_output, states, scale = self.unroll_encoder(
				F, past_target, past_observed_values
			)

			samples = self.sample_decoder(F, past_target, states, scale)

			return samples

	class MyProbRNNEstimator(GluonEstimator):
		@validated()
		def __init__(
			self,
			prediction_length: int,
			context_length: int,
			freq: str,
			distr_output: DistributionOutput,
			num_cells: int,
			num_layers: int,
			num_sample_paths: int = 100,
			scaling: bool = True,
			batch_size: int = 32,
			trainer: Trainer = Trainer()
		) -> None:
			super().__init__(trainer=trainer, batch_size=batch_size)
			self.prediction_length = prediction_length
			self.context_length = context_length
			self.freq = freq
			self.distr_output = distr_output
			self.num_cells = num_cells
			self.num_layers = num_layers
			self.num_sample_paths = num_sample_paths
			self.scaling = scaling

		def create_transformation(self):
			# Feature transformation that the model uses for input.
			return AddObservedValuesIndicator(
				target_field=FieldName.TARGET,
				output_field=FieldName.OBSERVED_VALUES,
			)

		def create_training_data_loader(self, dataset, **kwargs):
			instance_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=ExpectedNumInstanceSampler(
					num_instances=1,
					min_future=self.prediction_length,
				),
				past_length=self.context_length + 1,
				future_length=self.prediction_length,
				time_series_fields=[
					FieldName.FEAT_DYNAMIC_REAL,
					FieldName.OBSERVED_VALUES,
				],
			)
			input_names = get_hybrid_forward_input_names(MyProbTrainRNN)
			return TrainDataLoader(
				dataset=dataset,
				transform=instance_splitter + SelectFields(input_names),
				batch_size=self.batch_size,
				stack_fn=functools.partial(batchify, ctx=self.trainer.ctx, dtype=self.dtype),
				decode_fn=functools.partial(as_in_context, ctx=self.trainer.ctx),
				**kwargs,
			)

		def create_training_network(self) -> MyProbTrainRNN:
			return MyProbTrainRNN(
				prediction_length=self.prediction_length,
				context_length=self.context_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_layers=self.num_layers,
				num_sample_paths=self.num_sample_paths,
				scaling=self.scaling
			)

		def create_predictor(
			self, transformation: Transformation, trained_network: mx.gluon.HybridBlock
		) -> Predictor:
			prediction_splitter = InstanceSplitter(
				target_field=FieldName.TARGET,
				is_pad_field=FieldName.IS_PAD,
				start_field=FieldName.START,
				forecast_start_field=FieldName.FORECAST_START,
				instance_sampler=TestSplitSampler(),
				past_length=self.context_length + 1,
				future_length=self.prediction_length,
				time_series_fields=[
					FieldName.FEAT_DYNAMIC_REAL,
					FieldName.OBSERVED_VALUES,
				],
			)
			prediction_network = MyProbPredRNN(
				prediction_length=self.prediction_length,
				context_length=self.context_length,
				distr_output=self.distr_output,
				num_cells=self.num_cells,
				num_layers=self.num_layers,
				num_sample_paths=self.num_sample_paths,
				scaling=self.scaling
			)

			copy_parameters(trained_network, prediction_network)

			return RepresentableBlockPredictor(
				input_transform=transformation + prediction_splitter,
				prediction_net=prediction_network,
				batch_size=self.trainer.batch_size,
				freq=self.freq,
				prediction_length=self.prediction_length,
				ctx=self.trainer.ctx,
			)

	estimator = MyProbRNNEstimator(
		prediction_length=24,
		context_length=48,
		freq="1H",
		num_cells=40,
		num_layers=2,
		distr_output=GaussianOutput(),
		trainer=Trainer(
			ctx="cpu",
			epochs=5,
			learning_rate=1e-3,
			hybridize=False,
			num_batches_per_epoch=100
		)
	)

	predictor = estimator.train(train_ds)

	forecast_it, ts_it = make_evaluation_predictions(
		dataset=test_ds,  # Test dataset.
		predictor=predictor,  # Predictor.
		num_samples=100,  # Number of sample paths we want for evaluation.
	)

	forecasts = list(forecast_it)
	tss = list(ts_it)

	plot_prob_forecasts(tss[0], forecasts[0])

def r_forecast_package():
	import ast
	from gluonts.model.r_forecast import RForecastPredictor

	dataset = get_dataset("exchange_rate", regenerate=False)

	prediction_length = dataset.metadata.prediction_length
	freq = dataset.metadata.freq
	cardinality = ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
	train_ds = dataset.train
	test_ds = dataset.test

	#--------------------

	# ETS.
	ets_predictor = RForecastPredictor(
		freq=freq, 
		prediction_length=prediction_length, 
		method_name="ets", 
	)

	ets_forecast = list(ets_predictor.predict(train_ds))

	# ARIMA.
	arima_predictor = RForecastPredictor(
		freq=freq, 
		prediction_length=prediction_length, 
		method_name="arima", 
	)

	arima_forecast = list(arima_predictor.predict(train_ds))

def deepar_test():
	url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
	df = pd.read_csv(url, header=0, index_col=0)
	data = ListDataset(
		[{
			"start": df.index[0],
			"target": df.value[:"2015-04-05 00:00:00"]
		}],
		freq="5min"
	)

	trainer = Trainer(epochs=10)
	estimator = DeepAREstimator(freq="5min", prediction_length=12, trainer=trainer)
	predictor = estimator.train(training_data=data)

	prediction = next(predictor.predict(data))
	print(prediction.mean)
	prediction.plot(output_file="./graph.png")

# REF [site] >> https://ts.gluon.ai/stable/api/gluonts/gluonts.model.prophet.html
def prophet_test():
	raise NotImplementedError

# REF [site] >> https://ts.gluon.ai/stable/api/gluonts/gluonts.model.tft.html
def temporal_fusion_transformer_test():
	raise NotImplementedError

def npts_test():
	import ast
	from gluonts.model.npts import NPTSPredictor

	dataset = get_dataset("exchange_rate", regenerate=False)

	prediction_length = dataset.metadata.prediction_length
	freq = dataset.metadata.freq
	cardinality = ast.literal_eval(dataset.metadata.feat_static_cat[0].cardinality)
	train_ds = dataset.train
	test_ds = dataset.test

	npts_predictor = NPTSPredictor(freq=freq, prediction_length=prediction_length, context_length=300, kernel_type="uniform", use_seasonal_model=False)

	npts_forecast = list(npts_predictor.predict(train_ds))

def main():
	#quick_start_tutorial()
	#extended_forecasting_tutorial()

	#r_forecast_package()
	#deepar_test()
	#prophet_test()  # Not yet implemented.
	#temporal_fusion_transformer_test()  # Not yet implemented.
	npts_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
