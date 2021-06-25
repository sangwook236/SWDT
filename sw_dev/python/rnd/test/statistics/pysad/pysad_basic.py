#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://github.com/selimfirat/pysad
#	https://pysad.readthedocs.io/en/latest/

import numpy as np
from pysad.evaluation import AUROCMetric
from pysad.models import LODA, xStream
from pyod.models.iforest import IForest
from pysad.models.integrations import ReferenceWindowModel
from pysad.utils import ArrayStreamer
from pysad.transform.postprocessing import RunningAveragePostprocessor
from pysad.transform.preprocessing import InstanceUnitNormScaler
from pysad.transform.ensemble import AverageScoreEnsembler
from pysad.transform.probability_calibration import ConformalProbabilityCalibrator
from pysad.statistics import AverageMeter, VarianceMeter
from pysad.utils import ArrayStreamer, Data
from sklearn.utils import shuffle
from tqdm import tqdm

# REF [site] >> https://pysad.readthedocs.io/en/latest/examples.html
def full_usage_example():
	np.random.seed(61)  # Fix random seed.

	# Get data to stream.
	data = Data("data")
	X_all, y_all = data.get_data("arrhythmia.mat")
	X_all, y_all = shuffle(X_all, y_all)

	iterator = ArrayStreamer(shuffle=False)  # Init streamer to simulate streaming data.

	model = xStream()  # Init xStream anomaly detection model.
	preprocessor = InstanceUnitNormScaler()  # Init normalizer.
	postprocessor = RunningAveragePostprocessor(window_size=5)  # Init running average postprocessor.
	auroc = AUROCMetric()  # Init area under receiver-operating- characteristics curve metric.

	for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):  # Stream data.
		X = preprocessor.fit_transform_partial(X)  # Fit preprocessor to and transform the instance.

		score = model.fit_score_partial(X)  # Fit model to and score the instance.
		score = postprocessor.fit_transform_partial(score)  # Apply running averaging to the score.

		auroc.update(y, score)  # Update AUROC metric.

	# Output resulting AUROCS metric.
	print("AUROC: {}.".format(auroc.get()))

# REF [site] >> https://pysad.readthedocs.io/en/latest/examples.html
def statistics_usage_example():
	# Init data with mean 0 and standard deviation 1.
	X = np.random.randn(1000)

	# Init statistics trackers for mean and variance.
	avg_meter = AverageMeter()
	var_meter = VarianceMeter()

	for i in range(1000):
		# Update statistics trackers.
		avg_meter.update(X[i])
		var_meter.update(X[i])

	# Output resulting statistics.
	print(f"Average: {avg_meter.get()}, Standard deviation: {np.sqrt(var_meter.get())}")
	# It is close to random normal distribution with mean 0 and std 1 as we init the array via np.random.rand.

# REF [site] >> https://pysad.readthedocs.io/en/latest/examples.html
def ensembler_usage_example():
	np.random.seed(61)  # Fix random seed.

	data = Data("data")
	X_all, y_all = data.get_data("arrhythmia.mat")  # Load Aryhytmia data.
	X_all, y_all = shuffle(X_all, y_all)  # Shuffle data.
	iterator = ArrayStreamer(shuffle=False)  # Create streamer to simulate streaming data.
	auroc = AUROCMetric()  # Tracker of area under receiver-operating- characteristics curve metric.

	# Models to be ensembled.
	models = [
		xStream(),
		LODA()
	]
	ensembler = AverageScoreEnsembler()  # Ensembler module.

	for X, y in tqdm(iterator.iter(X_all, y_all)):  # Iterate over examples.
		model_scores = np.empty(len(models), dtype=np.float64)

		# Fit & Score via for each model.
		for i, model in enumerate(models):
			model.fit_partial(X)
			model_scores[i] = model.score_partial(X)

		score = ensembler.fit_transform_partial(model_scores)  # Fit to ensembler model and get ensembled score.

		auroc.update(y, score)  # Update AUROC metric.

	# Output score.
	print("AUROC: {}.".format(auroc.get()))

# REF [site] >> https://pysad.readthedocs.io/en/latest/examples.html
def probability_calibrator_usage_example():
	np.random.seed(61)  # Fix seed.

	model = xStream()  # Init model.
	calibrator = ConformalProbabilityCalibrator(windowed=True, window_size=300)  # Init probability calibrator.
	streaming_data = Data().get_iterator("arrhythmia.mat")  # Get streamer.

	for i, (x, y_true) in enumerate(streaming_data):  # Stream data.
		anomaly_score = model.fit_score_partial(x)  # Fit to an instance x and score it.

		calibrated_score = calibrator.fit_transform(anomaly_score)  # Fit & calibrate score.

		# Output if the instance is anomalous.
		if calibrated_score > 0.95:  # If probability of being normal is less than 5%.
			print(f"Alert: {i}th data point is anomalous.")

# REF [site] >> https://pysad.readthedocs.io/en/latest/examples.html
def PyOD_integration_example():
	np.random.seed(61)  # Fix seed.

	# Get data to stream.
	data = Data("data")
	X_all, y_all = data.get_data("arrhythmia.mat")
	X_all, y_all = shuffle(X_all, y_all)
	iterator = ArrayStreamer(shuffle=False)

	# Fit reference window integration to first 100 instances initially.
	model = ReferenceWindowModel(model_cls=IForest, window_size=240, sliding_size=30, initial_window_X=X_all[:100])

	auroc = AUROCMetric()  # Init area under receiver-operating-characteristics curve metric tracker.

	for X, y in tqdm(iterator.iter(X_all[100:], y_all[100:])):

		model.fit_partial(X)  # Fit to the instance.
		score = model.score_partial(X)  # Score the instance.

		auroc.update(y, score)  # Update the metric.

	# Output AUROC metric.
	print("AUROC: {}.".format(auroc.get()))

def main():
	full_usage_example()
	#statistics_usage_example()
	#ensembler_usage_example()
	#probability_calibrator_usage_example()
	#PyOD_integration_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
