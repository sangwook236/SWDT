#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import pytorch_forecasting as ptf

warnings.filterwarnings("ignore")

# REF [site] >> https://pytorch-forecasting.readthedocs.io/en/stable/getting-started.html
def getting_started():
	# Load data.
	data = ...

	# Define dataset.
	max_encoder_length = 36
	max_prediction_length = 6
	training_cutoff = "YYYY-MM-DD"  # Day for cutoff.

	training = TimeSeriesDataSet(
		data[lambda x: x.date < training_cutoff],
		time_idx= ...,
		target= ...,
		#weight="weight",
		group_ids=[ ... ],
		max_encoder_length=max_encoder_length,
		max_prediction_length=max_prediction_length,
		static_categoricals=[ ... ],
		static_reals=[ ... ],
		time_varying_known_categoricals=[ ... ],
		time_varying_known_reals=[ ... ],
		time_varying_unknown_categoricals=[ ... ],
		time_varying_unknown_reals=[ ... ],
	)

	# Create validation and training dataset.
	validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
	batch_size = 128
	train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
	val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

	# Define trainer with early stopping.
	early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
	lr_logger = pl.callbacks.LearningRateMonitor()
	trainer = pl.Trainer(
		max_epochs=100,
		gpus=0,
		gradient_clip_val=0.1,
		limit_train_batches=30,
		callbacks=[lr_logger, early_stop_callback],
	)

	# Create the model.
	tft = ptf.TemporalFusionTransformer.from_dataset(
		training,
		learning_rate=0.03,
		hidden_size=32,
		attention_head_size=1,
		dropout=0.1,
		hidden_continuous_size=16,
		output_size=7,
		loss=ptf.metrics.QuantileLoss(),
		log_interval=2,
		reduce_on_plateau_patience=4
	)
	print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

	# Find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1).
	res = trainer.tuner.lr_find(
		tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
	)

	print(f"Suggested learning rate: {res.suggestion()}")
	fig = res.plot(show=True, suggest=True)
	fig.show()

	# Fit the model.
	trainer.fit(
		tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
	)

	# Calculate baseline absolute error.
	actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
	baseline_predictions = Baseline().predict(val_dataloader)
	print("Mean absolute error = {}.".format(ptf.metrics.SMAPE()(baseline_predictions, actuals).item()))

# REF [site] >> https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
def demand_forecasting_with_Temporal_Fusion_Transformer_tutorial():
	# Our example is a demand forecast from the Stallion kaggle competition.
	#	https://www.kaggle.com/utathya/future-volume-prediction

	# Load data.
	#	First, we need to transform our time series into a pandas dataframe where each row can be identified with a time step and a time series.
	#	Fortunately, most datasets are already in this format.
	#	For this tutorial, we will use the Stallion dataset from Kaggle describing sales of various beverages.
	#	Our task is to make a six-month forecast of the sold volume by stock keeping units (SKU), that is products, sold by an agency, that is a store.
	#	There are about 21 000 monthly historic sales records.
	#	In addition to historic sales we have information about the sales price, the location of the agency, special days such as holidays, and volume sold in the entire industry.

	# 	The dataset is already in the correct format but misses some important features.
	#	Most importantly, we need to add a time index that is incremented by one for each time step.
	#	Further, it is beneficial to add date features, which in this case means extracting the month from the date record.
	from pytorch_forecasting.data.examples import get_stallion_data

	data = get_stallion_data()

	# Add time index.
	data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month
	data["time_idx"] -= data["time_idx"].min()

	# Add additional features.
	data["month"] = data.date.dt.month.astype(str).astype("category")  # Categories have be strings.
	data["log_volume"] = np.log(data.volume + 1e-8)
	data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
	data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")

	# We want to encode special days as one variable and thus need to first reverse one-hot encoding.
	special_days = [
		"easter_day",
		"good_friday",
		"new_year",
		"christmas",
		"labor_day",
		"independence_day",
		"revolution_day_memorial",
		"regional_games",
		"fifa_u_17_world_cup",
		"football_gold_cup",
		"beer_capital",
		"music_fest",
	]
	data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
	print(data.sample(10, random_state=521))
	print(data.describe())

	# Create dataset and dataloaders.
	#	The next step is to convert the dataframe into a PyTorch Forecasting TimeSeriesDataSet.
	#	Apart from telling the dataset which features are categorical vs continuous and which are static vs varying in time, we also have to decide how we normalise the data.
	#	Here, we standard scale each time series separately and indicate that values are always positive.
	#	Generally, the EncoderNormalizer, that scales dynamically on each encoder sequence as you train, is preferred to avoid look-ahead bias induced by normalisation.
	#	However, you might accept look-ahead bias if you are having troubles to find a reasonably stable normalisation, for example, because there are a lot of zeros in your data.
	#	Or you expect a more stable normalization in inference.
	#	In the later case, you ensure that you do not learn "weird" jumps that will not be present when running inference, thus training on a more realistic data set.
	max_prediction_length = 6
	max_encoder_length = 24
	training_cutoff = data["time_idx"].max() - max_prediction_length

	training = TimeSeriesDataSet(
		data[lambda x: x.time_idx <= training_cutoff],
		time_idx="time_idx",
		target="volume",
		group_ids=["agency", "sku"],
		min_encoder_length=max_encoder_length // 2,  # Keep encoder length long (as it is in the validation set).
		max_encoder_length=max_encoder_length,
		min_prediction_length=1,
		max_prediction_length=max_prediction_length,
		static_categoricals=["agency", "sku"],
		static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
		time_varying_known_categoricals=["special_days", "month"],
		variable_groups={"special_days": special_days},  # Group of categorical variables can be treated as one variable.
		time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
		time_varying_unknown_categoricals=[],
		time_varying_unknown_reals=[
			"volume",
			"log_volume",
			"industry_volume",
			"soda_volume",
			"avg_max_temp",
			"avg_volume_by_agency",
			"avg_volume_by_sku",
		],
		target_normalizer=ptf.data.GroupNormalizer(
			groups=["agency", "sku"], transformation="softplus"
		),  # Use softplus and normalize by group.
		add_relative_time_idx=True,
		add_target_scales=True,
		add_encoder_length=True,
	)

	# Create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series.
	validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

	# Create dataloaders for model.
	batch_size = 128  # Set this between 32 to 128.
	train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
	val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

	#-----
	# Create baseline model
	# Evaluating a Baseline model that predicts the next 6 months by simply repeating the last observed volume gives us a simle benchmark that we want to outperform.

	# Calculate baseline mean absolute error, i.e. predict next value as the last available value from the history.
	actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
	baseline_predictions = Baseline().predict(val_dataloader)
	print("Mean absolute error = {}.".format((actuals - baseline_predictions).abs().mean().item()))

	#-----
	# Train the Temporal Fusion Transformer.

	# Find optimal learning rate.
	# Prior to training, you can identify the optimal learning rate with the PyTorch Lightning learning rate finder.

	# Configure network and trainer.
	pl.seed_everything(42)
	trainer = pl.Trainer(
		gpus=0,
		# Clipping gradients is a hyperparameter and important to prevent divergance of the gradient for recurrent neural networks.
		gradient_clip_val=0.1,
	)

	tft = ptf.TemporalFusionTransformer.from_dataset(
		training,
		# Not meaningful for finding the learning rate but otherwise very important.
		learning_rate=0.03,
		hidden_size=16,  # Most important hyperparameter apart from learning rate.
		# Number of attention heads. Set to up to 4 for large datasets.
		attention_head_size=1,
		dropout=0.1,  # Between 0.1 and 0.3 are good values.
		hidden_continuous_size=8,  # Set to <= hidden_size.
		output_size=7,  # 7 quantiles by default.
		loss=ptf.metrics.QuantileLoss(),
		# Reduce learning rate if no improvement in validation loss after x epochs.
		reduce_on_plateau_patience=4,
	)
	print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

	# Find optimal learning rate.
	res = trainer.tuner.lr_find(
		tft,
		train_dataloaders=train_dataloader,
		val_dataloaders=val_dataloader,
		max_lr=10.0,
		min_lr=1e-6,
	)

	print(f"Suggested learning rate: {res.suggestion()}")
	fig = res.plot(show=True, suggest=True)
	fig.show()

	# For the TemporalFusionTransformer, the optimal learning rate seems to be slightly lower than the suggested one.
	# Further, we do not directly want to use the suggested learning rate because PyTorch Lightning sometimes can get confused by the noise at lower learning rates and suggests rates far too low.
	# Manual control is essential.
	# We decide to pick 0.03 as learning rate.

	# Train model.
	# If you have troubles training the model and get an error AttributeError: module 'tensorflow._api.v2.io.gfile' has no attribute 'get_filesystem', consider either uninstalling tensorflow or first execute:
	#	import tensorflow as tf
	#	import tensorboard as tb
	#	tf.io.gfile = tb.compat.tensorflow_stub.io.gfile.

	# Configure network and trainer.
	early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
	lr_logger = pl.callbacks.LearningRateMonitor()  # Log the learning rate.
	logger = pl.loggers.TensorBoardLogger("lightning_logs")  # Logging results to a tensorboard.

	trainer = pl.Trainer(
		max_epochs=30,
		gpus=0,
		enable_model_summary=True,
		gradient_clip_val=0.1,
		limit_train_batches=30,  # Coment in for training, running valiation every 30 batches.
		#fast_dev_run=True,  # Comment in to check that networkor dataset has no serious bugs.
		callbacks=[lr_logger, early_stop_callback],
		logger=logger,
	)

	tft = ptf.TemporalFusionTransformer.from_dataset(
		training,
		learning_rate=0.03,
		hidden_size=16,
		attention_head_size=1,
		dropout=0.1,
		hidden_continuous_size=8,
		output_size=7,  # 7 quantiles by default.
		loss=ptf.metrics.QuantileLoss(),
		log_interval=10,  # Uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches.
		reduce_on_plateau_patience=4,
	)
	print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

	# Fit network.
	trainer.fit(
		tft,
		train_dataloaders=train_dataloader,
		val_dataloaders=val_dataloader,
	)

	# Hyperparameter tuning.
	# Hyperparamter tuning with optuna(https://optuna.org/) is directly build into pytorch-forecasting.
	# For example, we can use the optimize_hyperparameters() function to optimize the TFT's hyperparameters.
	from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

	# Create study.
	study = optimize_hyperparameters(
		train_dataloader,
		val_dataloader,
		model_path="optuna_test",
		n_trials=200,
		max_epochs=50,
		gradient_clip_val_range=(0.01, 1.0),
		hidden_size_range=(8, 128),
		hidden_continuous_size_range=(8, 128),
		attention_head_size_range=(1, 4),
		learning_rate_range=(0.001, 0.1),
		dropout_range=(0.1, 0.3),
		trainer_kwargs=dict(limit_train_batches=30),
		reduce_on_plateau_patience=4,
		use_learning_rate_finder=False,  # Use Optuna to find ideal learning rate or use in-built learning rate finder.
	)

	# Save study results - also we can resume tuning at a later point in time.
	with open("test_study.pkl", "wb") as fout:
		pickle.dump(study, fout)

	# Show best hyperparameters.
	print("Best hyperparameters: {}.".format(study.best_trial.params))

	#-----
	# Evaluate performance.

	# Load the best model according to the validation loss (given that we use early stopping, this is not necessarily the last epoch).
	best_model_path = trainer.checkpoint_callback.best_model_path
	best_tft = ptf.TemporalFusionTransformer.load_from_checkpoint(best_model_path)

	# After training, we can make predictions with predict().
	# The method allows very fine-grained control over what it returns so that, for example, you can easily match predictions to your pandas dataframe.

	# Calcualte mean absolute error on validation set.
	actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
	predictions = best_tft.predict(val_dataloader)
	print("Mean absolute error = {}.".format((actuals - predictions).abs().mean().item()))

	# We can now also look at sample predictions directly which we plot with plot_prediction().
	# As you can see from the figures below, forecasts look rather accurate.
	# If you wonder, the grey lines denote the amount of attention the model pays to different points in time when making the prediction.
	# This is a special feature of the Temporal Fusion Transformer.

	# Raw predictions are a dictionary from which all kind of information including quantiles can be extracted.
	raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)

	for idx in range(10):  # Plot 10 examples.
		best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)

	# Worst performers.
	# Looking at the worst performers, for example in terms of SMAPE, gives us an idea where the model has issues with forecasting reliably.
	# These examples can provide important pointers about how to improve the model.
	# This kind of actuals vs predictions plots are available to all models.
	# Of course, it is also sensible to employ additional metrics, such as MASE, defined in the metrics module.
	# However, for the sake of demonstration, we only use SMAPE here.

	# Calcualte metric by which to display.
	predictions = best_tft.predict(val_dataloader)
	mean_losses = ptf.metrics.SMAPE(reduction="none")(predictions, actuals).mean(1)
	indices = mean_losses.argsort(descending=True)  # sort losses
	for idx in range(10):  # Plot 10 examples.
		best_tft.plot_prediction(
			x, raw_predictions, idx=indices[idx], add_loss_to_title=ptf.metrics.SMAPE(quantiles=best_tft.loss.quantiles)
		)

	# Actuals vs predictions by variables.
	# Checking how the model performs across different slices of the data allows us to detect weaknesses.
	# Plotted below are the means of predictions vs actuals across each variable divided into 100 bins using the Now, we can directly predict on the generated data using the calculate_prediction_actual_by_variable() and plot_prediction_actual_by_variable() methods.
	# The gray bars denote the frequency of the variable by bin, i.e. are a histogram.
	predictions, x = best_tft.predict(val_dataloader, return_x=True)
	predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
	best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

	#-----
	# Predict on selected data.
	# To predict on a subset of data we can filter the subsequences in a dataset using the filter() method.
	# Here we predict for the subsequence in the training dataset that maps to the group ids "Agency_01" and "SKU_01" and whose first predicted value corresponds to the time index "15".
	# We output all seven quantiles.
	# This means we expect a tensor of shape 1 x n_timesteps x n_quantiles = 1 x 6 x 7 as we predict for a single subsequence six time steps ahead and 7 quantiles for each time step.
	best_tft.predict(
		training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
		mode="quantiles",
	)

	# Of course, we can also plot this prediction readily.
	raw_prediction, x = best_tft.predict(
		training.filter(lambda x: (x.agency == "Agency_01") & (x.sku == "SKU_01") & (x.time_idx_first_prediction == 15)),
		mode="raw",
		return_x=True,
	)
	best_tft.plot_prediction(x, raw_prediction, idx=0)

	#-----
	# Predict on new data.
	# Because we have covariates in the dataset, predicting on new data requires us to define the known covariates upfront.

	# Select last 24 months from data (max_encoder_length is 24).
	encoder_data = data[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

	# Select last known data point and create decoder data from it by repeating it and incrementing the month
	# in a real world dataset, we should not just forward fill the covariates but specify them to account
	# for changes in special days and prices (which you absolutely should do but we are too lazy here).
	last_data = data[lambda x: x.time_idx == x.time_idx.max()]
	decoder_data = pd.concat(
		[last_data.assign(date=lambda x: x.date + pd.offsets.MonthBegin(i)) for i in range(1, max_prediction_length + 1)],
		ignore_index=True,
	)

	# Add time index consistent with "data".
	decoder_data["time_idx"] = decoder_data["date"].dt.year * 12 + decoder_data["date"].dt.month
	decoder_data["time_idx"] += encoder_data["time_idx"].max() + 1 - decoder_data["time_idx"].min()

	# Adjust additional time feature(s).
	decoder_data["month"] = decoder_data.date.dt.month.astype(str).astype("category")  # Categories have be strings.

	# Combine encoder and decoder data.
	new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

	# Now, we can directly predict on the generated data using the predict() method.
	new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

	for idx in range(10):  # Plot 10 examples.
		best_tft.plot_prediction(new_x, new_raw_predictions, idx=idx, show_future_observed=False)

	#-----
	# Interpret model.

	# Variable importances.
	# The model has inbuilt interpretation capabilities due to how its architecture is build.
	# Let's see how that looks.
	# We first calculate interpretations with interpret_output() and plot them subsequently with plot_interpretation().
	interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
	best_tft.plot_interpretation(interpretation)

	# Unsurprisingly, the past observed volume features as the top variable in the encoder and price related variables are among the top predictors in the decoder.
	# The general attention patterns seems to be that more recent observations are more important and older ones.
	# This confirms intuition.
	# The average attention is often not very useful - looking at the attention by example is more insightful because patterns are not averaged out.

	# Partial dependency.
	# Partial dependency plots are often used to interpret the model better (assuming independence of features).
	# They can be also useful to understand what to expect in case of simulations and are created with predict_dependency().
	dependency = best_tft.predict_dependency(
		val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
	)

	# Plotting median and 25% and 75% percentile.
	agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(
		median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
	)
	ax = agg_dependency.plot(y="median")
	ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3)

# REF [site] >> https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/building.html
def how_to_use_custom_data_and_implement_custom_models_and_metrics_tutorial():
	raise NotImplementedError

# REF [site] >> https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html
def autoregressive_modelling_with_DeepAR_and_DeepVAR_tutorial():
	# Load data.
	# The data consists of a quadratic trend and a seasonality component.
	from pytorch_forecasting.data.examples import generate_ar_data

	data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
	data["static"] = 2
	data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
	print(data.head())

	data = data.astype(dict(series=str))

	# Create dataset and dataloaders.
	max_encoder_length = 60
	max_prediction_length = 20

	training_cutoff = data["time_idx"].max() - max_prediction_length

	context_length = max_encoder_length
	prediction_length = max_prediction_length

	training = ptf.TimeSeriesDataSet(
		data[lambda x: x.time_idx <= training_cutoff],
		time_idx="time_idx",
		target="value",
		categorical_encoders={"series": ptf.data.NaNLabelEncoder().fit(data.series)},
		group_ids=["series"],
		static_categoricals=[
			"series"
		],  # As we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier).
		time_varying_unknown_reals=["value"],
		max_encoder_length=context_length,
		max_prediction_length=prediction_length,
	)

	validation = ptf.TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
	batch_size = 128
	# Synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR.
	train_dataloader = training.to_dataloader(
		train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
	)
	val_dataloader = validation.to_dataloader(
		train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
	)

	#-----
	# Calculate baseline error
	# Our baseline model predicts future values by repeating the last know value.
	# The resulting SMAPE is disappointing and should be easy to beat.

	pl.seed_everything(42)

	# Calculate baseline absolute error.
	actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
	baseline_predictions = ptf.Baseline().predict(val_dataloader)
	print("Mean absolute error = {}.".format(ptf.metrics.SMAPE()(baseline_predictions, actuals).item()))

	trainer = pl.Trainer(gpus=0, gradient_clip_val=1e-1)
	# The DeepAR model can be easily changed to a DeepVAR model by changing the applied loss function to a multivariate one, e.g. MultivariateNormalDistributionLoss.
	net = ptf.DeepAR.from_dataset(
		training,
		learning_rate=3e-2,
		hidden_size=30,
		rnn_layers=2,
		# FIXME [error] >> AttributeError: module 'pytorch_forecasting.metrics' has no attribute 'MultivariateNormalDistributionLoss'.
		loss=ptf.metrics.MultivariateNormalDistributionLoss(rank=30),
	)

	#-----
	# Train network.
	# Finding the optimal learning rate using PyTorch Lightning is easy.

	# Find optimal learning rate.
	res = trainer.tuner.lr_find(
		net,
		train_dataloaders=train_dataloader,
		val_dataloaders=val_dataloader,
		min_lr=1e-5,
		max_lr=1e0,
		early_stop_threshold=100,
	)
	print(f"Suggested learning rate: {res.suggestion()}")
	fig = res.plot(show=True, suggest=True)
	fig.show()
	net.hparams.learning_rate = res.suggestion()
	plt.show()

	early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
	trainer = pl.Trainer(
		max_epochs=30,
		gpus=0,
		enable_model_summary=True,
		gradient_clip_val=0.1,
		callbacks=[early_stop_callback],
		limit_train_batches=50,
		enable_checkpointing=True,
	)

	net = ptf.DeepAR.from_dataset(
		training,
		learning_rate=0.1,
		log_interval=10,
		log_val_interval=1,
		hidden_size=30,
		rnn_layers=2,
		# FIXME [error] >> AttributeError: module 'pytorch_forecasting.metrics' has no attribute 'MultivariateNormalDistributionLoss'.
		#loss=ptf.metrics.MultivariateNormalDistributionLoss(rank=30),
	)

	trainer.fit(
		net,
		train_dataloaders=train_dataloader,
		val_dataloaders=val_dataloader,
	)

	best_model_path = trainer.checkpoint_callback.best_model_path
	best_model = ptf.DeepAR.load_from_checkpoint(best_model_path)

	actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
	predictions = best_model.predict(val_dataloader)
	print("Mean absolute error = {}.".format((actuals - predictions).abs().mean().item()))

	raw_predictions, x = net.predict(val_dataloader, mode="raw", return_x=True, n_samples=100)

	series = validation.x_to_index(x)["series"]
	for idx in range(20):  # Plot 20 examples.
		# FIXME [error] >> ValueError: Expected parameter scale (Tensor of shape (100, 20)) of distribution Normal(loc: torch.Size([100, 20]), scale: torch.Size([100, 20])) to satisfy the constraint GreaterThan(lower_bound=0.0), but found invalid values.
		best_model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
		plt.suptitle(f"Series: {series.iloc[idx]}")
	plt.show()

	# When using DeepVAR as a multivariate forecaster, we might be also interested in the correlation matrix.
	# Here, there is no correlation between the series and we probably would need to train longer for this to show up.
	cov_matrix = best_model.loss.map_x_to_distribution(
		best_model.predict(val_dataloader, mode=("raw", "prediction"), n_samples=None)
	).base_dist.covariance_matrix.mean(0)

	# Normalize the covariance matrix diagnoal to 1.0.
	correlation_matrix = cov_matrix / torch.sqrt(torch.diag(cov_matrix)[None] * torch.diag(cov_matrix)[None].T)

	fig, ax = plt.subplots(1, 1, figsize=(10, 10))
	ax.imshow(correlation_matrix, cmap="bwr")
	plt.show()

	# Distribution of off-diagonal correlations.
	plt.hist(correlation_matrix[correlation_matrix < 1].numpy())
	plt.show()

def main():
	#getting_started()  # Not working.

	# Tutorials.
	#demand_forecasting_with_Temporal_Fusion_Transformer_tutorial()
	#how_to_use_custom_data_and_implement_custom_models_and_metrics_tutorial()  # Not yet implemented.
	autoregressive_modelling_with_DeepAR_and_DeepVAR_tutorial()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
