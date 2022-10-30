#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import darts
import darts.models, darts.metrics, darts.datasets
#import darts.utils, darts.dataprocessing
import sklearn.linear_model, sklearn.gaussian_process

# REF [site] >> https://unit8co.github.io/darts/README.html
def example_usage():
	# Read a pandas DataFrame.
	# REF [site] >> https://github.com/selva86/datasets
	df = pd.read_csv("./AirPassengers.csv", delimiter=",")

	# Create a TimeSeries, specifying the time and value columns.
	series = darts.TimeSeries.from_dataframe(df, "Month", "#Passengers")

	# Set aside the last 36 months as a validation series.
	train, val = series[:-36], series[-36:]

	# Fit an exponential smoothing model, and make a (probabilistic) prediction over the validation series' duration.
	model = darts.models.ExponentialSmoothing()
	model.fit(train)
	prediction = model.predict(len(val), num_samples=1000)

	# Plot the median, 5th and 95th percentiles.
	series.plot()
	prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
	plt.legend()

# REF [site] >> https://unit8co.github.io/darts/quickstart/00-quickstart.html
def building_and_manipulating_time_series_example():
	# TimeSeries is the main data class in Darts.
	# A TimeSeries represents a univariate or multivariate time series, with a proper time index.
	# The time index can either be of type pandas.DatetimeIndex (containing datetimes), or of type pandas.RangeIndex (containing integers; useful for representing sequential data without specific timestamps).
	# In some cases, TimeSeries can even represent probabilistic series, in order for instance to obtain confidence intervals.
	# All models in Darts consume TimeSeries and produce TimeSeries.

	# Read data and build a TimeSeries.
	# TimeSeries can be built easily using a few factory methods:
	#	From an entire Pandas DataFrame, using TimeSeries.from_dataframe().
	#	From a time index and an array of corresponding values, using TimeSeries.from_times_and_values().
	#	From a NumPy array of values, using TimeSeries.from_values().
	#	From a Pandas Series, using TimeSeries.from_series().
	#	From an xarray.DataArray, using TimeSeries.from_xarray().
	#	From a CSV file, using TimeSeries.from_csv().

	series = darts.datasets.AirPassengersDataset().load()
	series.plot()
	plt.show()

	#-----
	# Some TimeSeries operations.

	# Splitting.
	series1, series2 = series.split_before(0.75)
	series1.plot()
	series2.plot()
	plt.show()

	# Slicing.
	series1, series2 = series[:-36], series[-36:]
	series1.plot()
	series2.plot()
	plt.show()

	# Arithmetic operations.
	series_noise = darts.TimeSeries.from_times_and_values(
		series.time_index, np.random.randn(len(series))
	)
	(series / 2 + 20 * series_noise - 10).plot()
	plt.show()

	# Stacking.
	(series / 50).stack(series_noise).plot()
	plt.show()

	# Mapping.
	series.map(np.log).plot()
	plt.show()
	# Mapping on both timestamps and values.
	series.map(lambda ts, x: x / ts.days_in_month).plot()
	plt.show()

	# Adding some datetime attribute as an extra dimension (yielding a multivariate series).
	(series / 20).add_datetime_attribute("month").plot()
	plt.show()
	# Adding some binary holidays component.
	(series / 200).add_holidays("US").plot()
	plt.show()

	# Differencing.
	series.diff().plot()
	plt.show()

	# Filling missing values (using a "utils" function).
	# Missing values are represented by np.nan.
	values = np.arange(50, step=0.5)
	values[10:30] = np.nan
	values[60:95] = np.nan
	series_ = darts.TimeSeries.from_values(values)

	(series_ - 10).plot(label="with missing values (shifted below)")
	darts.utils.missing_values.fill_missing_values(series_).plot(label="without missing values")
	plt.show()

	#-----
	# Creating a training and validation series.
	train, val = series.split_before(pd.Timestamp("19580101"))
	train.plot(label="training")
	val.plot(label="validation")
	plt.show()

# REF [site] >> https://unit8co.github.io/darts/quickstart/00-quickstart.html
def training_forecasting_models_and_making_predictions_example():
	series = darts.datasets.AirPassengersDataset().load()
	#series.plot()
	#plt.show()

	train, val = series.split_before(pd.Timestamp("19580101"))
	#train.plot(label="training")
	#val.plot(label="validation")
	#plt.show()

	#-----
	# Playing with toy models.
	# There is a collection of "naive" baseline models in Darts, which can be very useful to get an idea of the bare minimum accuracy that one could expect.
	# For example, the NaiveSeasonal(K) model always "repeats" the value that occured K time steps ago.
	naive_model = darts.models.NaiveSeasonal(K=1)
	naive_model.fit(train)
	naive_forecast = naive_model.predict(36)

	series.plot(label="actual")
	naive_forecast.plot(label="naive forecast (K=1)")
	plt.show()

	# Inspect seasonality.
	# We can already improve by exploiting the seasonality in the data.
	# It seems quite obvious that the data has a yearly seasonality, which we can confirm by looking at the auto-correlation function (ACF), and highlighting the lag m=12.
	darts.utils.statistics.plot_acf(train, m=12, alpha=0.05)
	plt.show()

	# The ACF presents a spike at x = 12, which suggests a yearly seasonality trend (highlighted in red).
	# The blue zone determines the significance of the statistics for a confidence level of alpha = 5%.
	# We can also run a statistical check of seasonality for each candidate period m.
	for m in range(2, 25):
		is_seasonal, period = darts.utils.statistics.check_seasonality(train, m=m, alpha=0.05)
		if is_seasonal:
			print("There is seasonality of order {}.".format(period))

	# A less naive model.
	# Let's try the NaiveSeasonal model again with a seasonality of 12.
	seasonal_model = darts.models.NaiveSeasonal(K=12)
	seasonal_model.fit(train)
	seasonal_forecast = seasonal_model.predict(36)

	series.plot(label="actual")
	seasonal_forecast.plot(label="naive forecast (K=12)")
	plt.show()

	# We are still missing the trend.
	# Fortunately, there is also another naive baseline model capturing the trend, which is called NaiveDrift.
	# This model simply produces linear predictions, with a slope that is determined by the first and last values of the training set.
	drift_model = darts.models.NaiveDrift()
	drift_model.fit(train)
	drift_forecast = drift_model.predict(36)

	combined_forecast = drift_forecast + seasonal_forecast - train.last_value()

	series.plot()
	combined_forecast.plot(label="combined")
	drift_forecast.plot(label="drift")
	plt.show()

	#-----
	# Computing error metrics.

	# Mean Absolute Percentage Error (MAPE).
	print(
		"Mean absolute percentage error for the combined naive drift + seasonal: {:.2f}%.".format(
			darts.metrics.mape(series, combined_forecast)
		)
	)

	# darts.metrics contains many more metrics to compare time series.
	# The metrics will compare only common slices of series when the two series are not aligned, and parallelize computation over a large number of pairs of series.

	#-----
	# Quickly try out several models.
	# Darts is built to make it easy to train and validate several models in a unified way.
	def eval_model(model):
		model.fit(train)
		forecast = model.predict(len(val))
		print("model {} obtains MAPE: {:.2f}%".format(model, darts.metrics.mape(val, forecast)))

	eval_model(darts.models.ExponentialSmoothing())
	eval_model(darts.models.TBATS())
	eval_model(darts.models.AutoARIMA())
	eval_model(darts.models.Theta())

	# Searching for hyper-parameters with the Theta method.
	# Search for the best theta parameter, by trying 50 different values.
	thetas = 2 - np.linspace(-10, 10, 50)

	best_mape = float("inf")
	best_theta = 0

	for theta in thetas:
		model = darts.models.Theta(theta)
		model.fit(train)
		pred_theta = model.predict(len(val))
		res = darts.metrics.mape(val, pred_theta)

		if res < best_mape:
			best_mape = res
			best_theta = theta

	best_theta_model = darts.models.Theta(best_theta)
	best_theta_model.fit(train)
	pred_best_theta = best_theta_model.predict(len(val))

	print(
		"The MAPE is: {:.2f}, with theta = {}.".format(
			darts.metrics.mape(val, pred_best_theta), best_theta
		)
	)

	train.plot(label="train")
	val.plot(label="true")
	pred_best_theta.plot(label="prediction")
	plt.show()

	#-----
	# Backtesting: simulate historical forecasting
	# Backtesting simulates predictions that would have been obtained historically with a given model.
	# It can take a while to produce, since the model is (by default) re-trained every time the simulated prediction time advances.

	# Such simulated forecasts are always defined with respect to a forecast horizon, which is the number of time steps that separate the prediction time from the forecast time.
	# In the example below, we simulate forecasts done for 3 months in the future (compared to prediction time).
	historical_fcast_theta = best_theta_model.historical_forecasts(
		series, start=0.6, forecast_horizon=3, verbose=True
	)

	series.plot(label="data")
	historical_fcast_theta.plot(label="backtest 3-months ahead forecast (Theta)")
	print("MAPE = {:.2f}%".format(darts.metrics.mape(historical_fcast_theta, series)))
	plt.show()

	# To have a closer look at the errors, we can also use the backtest() method to obtain all the raw errors (say, MAPE errors) that would have been obtained by our model.
	best_theta_model = darts.models.Theta(best_theta)

	raw_errors = best_theta_model.backtest(
		series, start=0.6, forecast_horizon=3, metric=darts.metrics.mape, reduction=None, verbose=True
	)

	darts.utils.statistics.plot_hist(
		raw_errors,
		bins=np.arange(0, max(raw_errors), 1),
		title="Individual backtest error scores (histogram)",
	)
	plt.show()

	# Using backtest() we can also get a simpler view of the average error over the historical forecasts.
	average_error = best_theta_model.backtest(
		series,
		start=0.6,
		forecast_horizon=3,
		metric=darts.metrics.mape,
		reduction=np.mean,  # This is actually the default.
		#reduction=np.median,  # Median MAPE.
		verbose=True,
	)

	print("Average error (MAPE) over all historical forecasts: %.2f" % average_error)

	# Let's look at the fitted value residuals of our current Theta model, i.e. the difference between the 1-step forecasts at every point in time obtained by fitting the model on all previous points, and the actual observed values.
	darts.utils.statistics.plot_residuals_analysis(best_theta_model.residuals(series))
	plt.show()

	# We can see that the distribution is not centered at 0, which means that our Theta model is biased.
	# We can also make out a large ACF value at lag equal to 12, which indicates that the residuals contain information that was not used by the model.

	# ExponentialSmoothing model.
	model_es = darts.models.ExponentialSmoothing()
	historical_fcast_es = model_es.historical_forecasts(series, start=0.6, forecast_horizon=3, verbose=True)

	series.plot(label="data")
	historical_fcast_es.plot(label="backtest 3-months ahead forecast (Exp. Smoothing)")
	print("MAPE = {:.2f}%".format(darts.metrics.mape(historical_fcast_es, series)))
	plt.show()

	# We get a mean absolute percentage error of about 4-5% when backtesting with a 3-months forecast horizon in this case.
	darts.utils.statistics.plot_residuals_analysis(model_es.residuals(series))
	plt.show()

	# The residual analysis also reflects an improved performance in that we now have a distribution of the residuals centred at value 0, and the ACF values, although not insignificant, have lower magnitudes.

# REF [site] >> https://unit8co.github.io/darts/quickstart/00-quickstart.html
def machine_learning_and_global_models_example():
	# Darts has a rich support for machine learning and deep learning forecasting models:
	#	RegressionModel can wrap around any sklearn-compatible regression model to produce forecasts.
	#	RNNModel is a flexible RNN implementation, which can be used like DeepAR.
	#	NBEATSModel implements the N-BEATS model.
	#	TFTModel implements the Temporal Fusion Transformer model.
	#	TCNModel implements temporal convolutional networks.
	#	...

	# In addition to supporting the same basic fit()/predict() interface as the other models, these models are also global models, as they support being trained on multiple time series (sometimes referred to as meta learning).
	# This is a key point of using ML-based models for forecasting: more often than not, ML models (especially deep learning models) need to be trained on large amounts of data, which often means a large amount of separate yet related time series.

	# A toy example with two series.
	series_air = darts.datasets.AirPassengersDataset().load().astype(np.float32)
	series_milk = darts.datasets.MonthlyMilkDataset().load().astype(np.float32)

	# Set aside last 36 months of each series as validation set.
	train_air, val_air = series_air[:-36], series_air[-36:]
	train_milk, val_milk = series_milk[:-36], series_milk[-36:]

	train_air.plot()
	val_air.plot()
	train_milk.plot()
	val_milk.plot()
	plt.show()

	# Let's scale these two series between 0 and 1, as that will benefit most ML models.
	scaler = darts.dataprocessing.transformers.Scaler()
	train_air_scaled, train_milk_scaled = scaler.fit_transform([train_air, train_milk])

	train_air_scaled.plot()
	train_milk_scaled.plot()
	plt.show()

	# We can also parallelize this sort of operations over multiple processors by specifying n_jobs.

	#-----
	if True:
		# Using deep learning: example with N-BEATS.
		model = darts.models.NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)

		model.fit([train_air_scaled, train_milk_scaled], epochs=50, verbose=True)

		# The output_chunk_length does not directly constrain the forecast horizon n that can be used with predict().
		# Here, we trained the model with output_chunk_length=12 and produce forecasts for n=36 months ahead; this is simply done in an auto-regressive way behind the scenes (where the network recursively consumes its previous outputs).
		pred_air = model.predict(series=train_air_scaled, n=36)
		pred_milk = model.predict(series=train_milk_scaled, n=36)

		# Scale back.
		pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

		plt.figure(figsize=(10, 6))
		series_air.plot(label="actual (air)")
		series_milk.plot(label="actual (milk)")
		pred_air.plot(label="forecast (air)")
		pred_milk.plot(label="forecast (milk)")
		plt.show()

	#-----
	# Covariates: using external data.
	# In addition to the target series (the series we are interested to forecast), many models in Darts also accept covariates series in input.
	# Covariates are series that we do not want to forecast, but which can provide helpful additional information to the models.
	# Both the targets and covariates can be multivariate or univariate.
	# There are two kinds of covariate time series in Darts:
	#	past_covariates are series not necessarily known ahead of the forecast time.
	#		Those can for instance represent things that have to be measured and are not known upfront.
	#		Models do not use the future values of past_covariates when making forecasts.
	#	future_covariates are series which are known in advance, up to the forecast horizon.
	#		This can represent things such as calendar information, holidays, weather forecasts, etc.
	#		Models that accept future_covariates will look at the future values (up to the forecast horizon) when making forecasts.

	# Each covariate can potentially be multivariate.
	# If you have several covariate series (such as month and year values), you should stack() or concatenate() them to obtain a multivariate series.
	# The covariates you provide can be longer than necessary.
	# Darts will try to be smart and slice them in the right way for forecasting the target, based on the time indexes of the different series.
	# You will receive an error if your covariates do not have a sufficient time span, though.

	from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

	air_covs = darts.concatenate(
		[
			dt_attr(series_air.time_index, "month", dtype=np.float32) / 12,
			(dt_attr(series_air.time_index, "year", dtype=np.float32) - 1948) / 12,
		],
		axis="component",
	)
	milk_covs = darts.concatenate(
		[
			dt_attr(series_milk.time_index, "month", dtype=np.float32) / 12,
			(dt_attr(series_milk.time_index, "year", dtype=np.float32) - 1962) / 13,
		],
		axis="component",
	)

	air_covs.plot()
	plt.title("one multivariate time series of 2 dimensions, containing covariates for the air series:")
	plt.show()

	if True:
		# Not all models support all types of covariates.
		# NBEATSModel supports only past_covariates.
		model = darts.models.NBEATSModel(input_chunk_length=24, output_chunk_length=12, random_state=42)

		model.fit(
			[train_air_scaled, train_milk_scaled],
			past_covariates=[air_covs, milk_covs],
			epochs=50,
			verbose=True,
		)

		# Then to produce forecasts, we again have to provide our covariates as past_covariates to the predict() function.
		# Even though the covariates time series also contains "future" values of the covariates up to the forecast horizon, the model will not consume those future values, because it uses them as past covariates (and not future covariates).
		pred_air = model.predict(series=train_air_scaled, past_covariates=air_covs, n=36)
		pred_milk = model.predict(series=train_milk_scaled, past_covariates=milk_covs, n=36)

		# Scale back.
		pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

		plt.figure(figsize=(10, 6))
		series_air.plot(label="actual (air)")
		series_milk.plot(label="actual (milk)")
		pred_air.plot(label="forecast (air)")
		pred_milk.plot(label="forecast (milk)")
		plt.show()

		# Encoders: using covariates for free.
		# Using covariates related to the calendar or time axis (such as months and years as in our example above) is so frequent that deep learning models in Darts have a built-in functionality to use such covariates out of the box.
		# To easily integrate such covariates to your model, you can simply specify the add_encoders parameter at model creation.
		# This parameter has to be a dictionary containing informations about what should be encoded as extra covariates.
		'''
		encoders = {
			"cyclic": {"future": ["month"]},
			"datetime_attribute": {"future": ["hour", "dayofweek"]},
			"position": {"past": ["absolute"], "future": ["relative"]},
			"custom": {"past": [lambda idx: (idx.year - 1950) / 50]},
			"transformer": darts.dataprocessing.transformers.Scaler(),
		}
		'''
		encoders = {
			"datetime_attribute": {"past": ["month", "year"]},
			"transformer": darts.dataprocessing.transformers.Scaler(),
		}

		model = darts.models.NBEATSModel(
			input_chunk_length=24,
			output_chunk_length=12,
			add_encoders=encoders,
			random_state=42,
		)

		model.fit([train_air_scaled, train_milk_scaled], epochs=50, verbose=True)

		pred_air = model.predict(series=train_air_scaled, n=36)

		# Scale back.
		pred_air = scaler.inverse_transform(pred_air)

		plt.figure(figsize=(10, 6))
		series_air.plot(label="actual (air)")
		pred_air.plot(label="forecast (air)")
		plt.show()

	#-----
	if True:
		# Regression forecasting models.
		# RegressionModel's are forecasting models which wrap around sklearn-compatible regression models.
		# The inner regression model is used to predict future values of the target series, as a function of certain lags of the target, past and future covariates.
		# By default, the RegressionModel will do a linear regression.
		# It is very easy to use any desired sklearn-compatible regression model by specifying the model parameter, but for convenience Darts also provides a couple of ready-made models out of the box:
		#	RandomForest wraps around sklearn.ensemble.RandomForestRegressor.
		#	LightGBMModel wraps around lightbm.
		#	LinearRegressionModel wraps around sklearn.linear_model.LinearRegression (accepting the same kwargs).

		model = darts.models.RegressionModel(lags=72, lags_future_covariates=[-6, 0], model=sklearn.linear_model.BayesianRidge())

		model.fit([train_air_scaled, train_milk_scaled], future_covariates=[air_covs, milk_covs])

		pred_air, pred_milk = model.predict(
			series=[train_air_scaled, train_milk_scaled],
			future_covariates=[air_covs, milk_covs],
			n=36,
		)

		# Scale back.
		pred_air, pred_milk = scaler.inverse_transform([pred_air, pred_milk])

		plt.figure(figsize=(10, 6))
		series_air.plot(label="actual (air)")
		series_milk.plot(label="actual (milk)")
		pred_air.plot(label="forecast (air)")
		pred_milk.plot(label="forecast (milk)")
		plt.show()

		# Metrics over sequences of series.
		print(darts.metrics.mape([series_air, series_milk], [pred_air, pred_milk]))
		# The average metric over "all" series.
		print(darts.metrics.mape([series_air, series_milk], [pred_air, pred_milk], inter_reduction=np.mean))
		# Computing metrics can be parallelized over N processors when executed over many series pairs by specifying n_jobs=N.

		# Backtest.
		bayes_ridge_model = darts.models.RegressionModel(lags=72, lags_future_covariates=[0], model=sklearn.linear_model.BayesianRidge())

		backtest = bayes_ridge_model.historical_forecasts(series_air, future_covariates=air_covs, start=0.6, forecast_horizon=3, verbose=True)

		print("MAPE = %.2f" % (darts.metrics.mape(backtest, series_air)))
		series_air.plot()
		backtest.plot()
		plt.show()

	#-----
	if True:
		# Probabilistic forecasts.
		# Some models can produce probabilistic forecasts.
		# This is the case for all deep learning models (such as RNNModel, NBEATSModel, etc.), as well as for ARIMA and ExponentialSmoothing.
		# The full list is available on https://github.com/unit8co/darts#forecasting-models.

		# For ARIMA and ExponentialSmoothing, one can simply specify a num_samples parameter to the predict() function.
		# The returned TimeSeries will then be composed of num_samples Monte Carlo samples describing the distribution of the time series' values.
		# The advantage of relying on Monte Carlo samples (in contrast to, say, explicit confidence intervals) is that they can be used to describe any parametric or non-parametric joint distribution over components, and compute arbitrary quantiles.
		model_es = darts.models.ExponentialSmoothing()
		model_es.fit(train_air)
		probabilistic_forecast = model_es.predict(len(val_air), num_samples=500)

		series_air.plot(label="actual")
		probabilistic_forecast.plot(label="probabilistic forecast")
		plt.legend()
		plt.show()

		# With neural networks.
		# With neural networks, one has to give a Likelihood object to the model.
		# The likelihoods specify which distribution the model will try to fit, along with potential prior values for the distributions' parameters.
		# The full list of available likelihoods is available in https://unit8co.github.io/darts/generated_api/darts.utils.likelihood_models.html.
		model = darts.models.TCNModel(
			input_chunk_length=24,
			output_chunk_length=12,
			random_state=42,
			likelihood=darts.utils.likelihood_models.LaplaceLikelihood(),
		)

		model.fit(train_air_scaled, epochs=400, verbose=True)

		# To get probabilistic forecasts, we again only need to specify some num_samples >> 1.
		pred = model.predict(n=36, num_samples=500)

		# Scale back.
		pred = scaler.inverse_transform(pred)

		series_air.plot()
		pred.plot()
		plt.show()

		# Furthermore, we could also for instance specify that we have some prior belief that the scale of the distribution is about 0.1 (in the transformed domain), while still capturing some time dependency of the distribution, by specifying prior_b=.1.
		# Behind the scenes this will regularize the training loss with a Kullback-Leibler divergence term.
		model = darts.models.TCNModel(
			input_chunk_length=24,
			output_chunk_length=12,
			random_state=42,
			likelihood=darts.utils.likelihood_models.LaplaceLikelihood(prior_b=0.1),
		)

		model.fit(train_air_scaled, epochs=400, verbose=True)

		pred = model.predict(n=36, num_samples=500)

		# Scale back.
		pred = scaler.inverse_transform(pred)

		series_air.plot()
		pred.plot()
		plt.show()

		# By default TimeSeries.plot() shows the median as well as the 5th and 95th percentiles (of the marginal distributions, if the TimeSeries is multivariate).
		pred.plot(low_quantile=0.01, high_quantile=0.99, label="1-99th percentiles")
		pred.plot(low_quantile=0.2, high_quantile=0.8, label="20-80th percentiles")
		plt.show()

		# Types of distributions.
		# The likelihood has to be compatible with the domain of your time series' values.
		# For instance PoissonLikelihood can be used on discrete positive values, ExponentialLikelihood can be used on real positive values, and BetaLikelihood on real values in (0, 1).
		# It is also possible to use QuantileRegression to apply a quantile loss and fit some desired quantiles directly.

		# Evaluating probabilistic forecasts.
		# How can we evaluate the quality of probabilistic forecasts?
		# By default, most metrics functions (such as mape()) will keep working but look only at the median forecast.
		# It is also possible to use the rho-risk metric (or quantile loss), which quantifies the error for each predicted quantiles.
		print("MAPE of median forecast: %.2f" % darts.metrics.mape(series_air, pred))
		for rho in [0.05, 0.1, 0.5, 0.9, 0.95]:
			rr = darts.metrics.rho_risk(series_air, pred, rho=rho)
			print("rho-risk at quantile %.2f: %.2f" % (rho, rr))

		# Using quantile loss.
		# Could we do better by fitting these quantiles directly?
		# We can just use a QuantileRegression likelihood.
		model = darts.models.TCNModel(
			input_chunk_length=24,
			output_chunk_length=12,
			random_state=42,
			likelihood=darts.utils.likelihood_models.QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]),
		)

		model.fit(train_air_scaled, epochs=400, verbose=True)

		pred = model.predict(n=36, num_samples=500)

		# Scale back.
		pred = scaler.inverse_transform(pred)

		series_air.plot()
		pred.plot()
		plt.show()

		print("MAPE of median forecast: %.2f" % darts.metrics.mape(series_air, pred))
		for rho in [0.05, 0.1, 0.5, 0.9, 0.95]:
			rr = darts.metrics.rho_risk(series_air, pred, rho=rho)
			print("rho-risk at quantile %.2f: %.2f" % (rho, rr))

	#-----
	if True:
		# Ensembling models.
		# Ensembling is about combining the forecasts produced by several models, in order to obtain a final - and hopefully better forecast.

		# Naive ensembling.
		# Naive ensembling just takes the average of the forecasts of several models.
		models = [darts.models.NaiveDrift(), darts.models.NaiveSeasonal(12)]

		ensemble_model = darts.models.NaiveEnsembleModel(models=models)

		backtest = ensemble_model.historical_forecasts(
			series_air, start=0.6, forecast_horizon=3, verbose=True
		)

		print("MAPE = %.2f" % (darts.metrics.mape(backtest, series_air)))
		series_air.plot()
		backtest.plot()
		plt.show()

		# Learned ensembling.
		# We can sometimes do better if we see the ensembling as a supervised regression problem: given a set of forecasts (features), find a model that combines them in order to minimise errors on the target.
		models = [darts.models.NaiveDrift(), darts.models.NaiveSeasonal(12)]

		ensemble_model = darts.models.RegressionEnsembleModel(forecasting_models=models, regression_train_n_points=12)

		backtest = ensemble_model.historical_forecasts(series_air, start=0.6, forecast_horizon=3, verbose=True)

		print("MAPE = %.2f" % (darts.metrics.mape(backtest, series_air)))
		series_air.plot()
		backtest.plot()
		plt.show()

		# We can also inspect the coefficients used to weigh the two inner models in the linear combination.
		print("Ensemble weights = {}.".format(ensemble_model.regression_model.model.coef_))

	#-----
	if True:
		# Filtering models.
		# In addition to forecasting models, which are able to predict future values of series, Darts also contains a couple of helpful filtering models, which can model "in sample" series' values distributions.

		# Fitting a Kalman filter.
		kf = darts.models.KalmanFilter(dim_x=3)
		kf.fit(train_air_scaled)
		filtered_series = kf.filter(train_air_scaled, num_samples=100)

		train_air_scaled.plot()
		filtered_series.plot()
		plt.show()

		# Inferring missing values with Gaussian Processes.
		# Darts also contains a GaussianProcessFilter which can be used for probabilistic modeling of series.
		# create a series with holes:
		values = train_air_scaled.values()
		values[20:22] = np.nan
		values[28:32] = np.nan
		values[55:59] = np.nan
		values[72:80] = np.nan
		series_holes = darts.TimeSeries.from_times_and_values(train_air_scaled.time_index, values)
		series_holes.plot()

		kernel = sklearn.gaussian_process.kernels.RBF()

		gpf = darts.models.GaussianProcessFilter(kernel=kernel, alpha=0.1, normalize_y=True)
		filtered_series = gpf.filter(series_holes, num_samples=100)

		filtered_series.plot()
		plt.show()

def main():
	example_usage()

	# Quickstart.
	#building_and_manipulating_time_series_example()
	#training_forecasting_models_and_making_predictions_example()
	#machine_learning_and_global_models_example()

	# User guide.
	# REF [site] >> https://unit8co.github.io/darts/userguide.html

	# Examples.
	# REF [site] >>
	#	https://unit8co.github.io/darts/examples.html
	#	https://github.com/unit8co/darts/tree/master/examples
	#
	#	Hyperparameter Optimization.
	#	Fast Fourier Transform.
	#	Dynamic Time Warping (DTW).
	#	Recurrent Neural Networks.
	#	Temporal Convolutional Networks.
	#	Transformer Model.
	#	N-BEATS Model.
	#	DeepAR Model.
	#	DeepTCN Model.
	#	Temporal Fusion Transformer (TFT) Model.
	#	Kalman Filter Model.
	#	Gaussian Process Filter Model.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
