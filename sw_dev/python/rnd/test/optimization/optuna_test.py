#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, logging
import numpy as np
import optuna

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html
def simple_quadratic_function_tutorial():
	def objective(trial):
		x = trial.suggest_float("x", -10, 10)
		return (x - 2)**2

	study = optuna.create_study()
	study.optimize(
		objective,
		n_trials=100,
		timeout=None,
		n_jobs=1,
		catch=(),
		callbacks=None,
		gc_after_trial=False,
		show_progress_bar=False,
	)

	print("Best params = {}.".format(study.best_params))

	best_params = study.best_params
	found_x = best_params["x"]
	print("Found x: {}, (x - 2)^2: {}.".format(found_x, (found_x - 2) ** 2))

	print("Best value = {}.".format(study.best_value))
	print("Best trial = {}.".format(study.best_trial))

	print("Trials = {}.".format(study.trials))
	print("#trials = {}.".format(len(study.trials)))

	# Continue the optimization.
	study.optimize(objective, n_trials=100)

	print("#trials = {}.".format(len(study.trials)))

	best_params = study.best_params
	found_x = best_params["x"]
	print("Found x: {}, (x - 2)^2: {}".format(found_x, (found_x - 2) ** 2))

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
def search_space_tutorial():
	if False:
		def objective(trial):
			# Categorical parameter.
			optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

			# Integer parameter.
			num_layers = trial.suggest_int("num_layers", 1, 3)

			# Integer parameter (log).
			num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

			# Integer parameter (discretized).
			num_units = trial.suggest_int("num_units", 10, 100, step=5)

			# Floating point parameter.
			dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

			# Floating point parameter (log).
			learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

			# Floating point parameter (discretized).
			drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
	elif True:
		# Branches.

		import sklearn.ensemble
		import sklearn.svm

		def objective(trial):
			classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
			if classifier_name == "SVC":
				svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
				classifier_obj = sklearn.svm.SVC(C=svc_c)
			else:
				rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
				classifier_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth)
	elif False:
		# Loops.

		import torch

		def create_model(trial, in_size):
			n_layers = trial.suggest_int("n_layers", 1, 3)

			layers = []
			for i in range(n_layers):
				n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
				layers.append(torch.nn.Linear(in_size, n_units))
				layers.append(torch.nn.ReLU())
				in_size = n_units
			layers.append(torch.nn.Linear(in_size, 10))

			return torch.nn.Sequential(*layers)

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
def efficient_optimization_algorithms_tutorial():
	# Sampling algorithms:
	#	GridSampler: Grid Search.
	#	RandomSampler: Random Search.
	#	TPESampler: Tree-structured Parzen Estimator algorithm.
	#	CmaEsSampler: CMA-ES based algorithm.
	#	PartialFixedSampler: Algorithm to enable partial fixed parameters.
	#	NSGAIISampler: Nondominated Sorting Genetic Algorithm II.
	#	QMCSampler: A Quasi Monte Carlo sampling algorithm.
	# The default sampler is TPESampler.

	study = optuna.create_study()
	print(f"Sampler is {study.sampler.__class__.__name__}")

	study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
	print(f"Sampler is {study.sampler.__class__.__name__}")

	study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
	print(f"Sampler is {study.sampler.__class__.__name__}")

	# Pruning algorithms:
	#	MedianPruner: Median pruning algorithm.
	#	NopPruner: Non-pruning algorithm.
	#	PatientPruner: Algorithm to operate pruner with tolerance.
	#	PercentilePruner: Algorithm to prune specified percentile of trials.
	#	SuccessiveHalvingPruner: Asynchronous Successive Halving algorithm.
	#	HyperbandPruner: Hyperband algorithm.
	#	ThresholdPruner: Threshold pruning algorithm.

	# To turn on the pruning feature, you need to call report() and should_prune() after each step of the iterative training.
	# report() periodically monitors the intermediate objective values. 
	# should_prune() decides termination of the trial that does not meet a predefined condition.

	import sklearn.datasets
	import sklearn.linear_model
	import sklearn.model_selection

	def objective(trial):
		iris = sklearn.datasets.load_iris()
		classes = list(set(iris.target))
		train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(
			iris.data, iris.target, test_size=0.25, random_state=0
		)

		alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
		clf = sklearn.linear_model.SGDClassifier(alpha=alpha)

		for step in range(100):
			clf.partial_fit(train_x, train_y, classes=classes)

			# Report intermediate objective value.
			intermediate_value = 1.0 - clf.score(valid_x, valid_y)
			trial.report(intermediate_value, step)

			# Handle pruning based on the intermediate value.
			if trial.should_prune():
				raise optuna.TrialPruned()

		return 1.0 - clf.score(valid_x, valid_y)

	# Add stream handler of stdout to show the messages.
	optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
	study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
	study.optimize(objective, n_trials=20)

	# Which sampler and pruner Should be used?
	#	REF [site] >> https://github.com/optuna/optuna/wiki/Benchmarks-with-Kurobako

	# For not deep learning tasks:
	#	For RandomSampler, MedianPruner is the best.
	#	For TPESampler, Hyperband is the best.
	# For deep learning tasks:
	#	Parallel Compute Resource	Categorical/Conditional Hyperparameters		Recommended Algorithms
	#	Limited						No											TPE. GP-EI if search space is low-dimensional and continuous.
	#								Yes											TPE. GP-EI if search space is low-dimensional and continuous
	#	Sufficient					No											CMA-ES, Random Search
	#								Yes											Random Search or Genetic Algorithm

	# Integration modules for pruning.
	#	Refer to xgboost_integration_example().

def easy_parallelization_tutorial():
	# Step 1. Create a study:
	#	Using optuna create-study command.
	#	Use optuna.create_study() in Python script.

	#$ mysql -u root -e "CREATE DATABASE IF NOT EXISTS example"
	#$ optuna create-study --study-name "distributed-example" --storage "mysql://root@localhost/example"

	# Step 2. Write an optimization script.

	def objective(trial):
		x = trial.suggest_float("x", -10, 10)
		return (x - 2)**2

	study = optuna.load_study(study_name="distributed-example", storage="mysql://root@localhost/example")
	study.optimize(objective, n_trials=100)

	# Step 3. Run the shared study from multiple processes.

	# Process 1:
	#	$ python foo.py
	# Process 2 (the same command as process 1):
	#	$ python foo.py

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/005_visualization.html
def quick_visualization_for_hyperparameter_optimization_analysis_tutorial():
	import lightgbm as lgb
	import sklearn.datasets
	import sklearn.metrics
	from sklearn.model_selection import train_test_split
	if False:
		# Plotly.
		from optuna.visualization import plot_contour
		from optuna.visualization import plot_edf
		from optuna.visualization import plot_intermediate_values
		from optuna.visualization import plot_optimization_history
		from optuna.visualization import plot_parallel_coordinate
		from optuna.visualization import plot_param_importances
		from optuna.visualization import plot_slice
	else:
		# Matplotlib.
		from optuna.visualization.matplotlib import plot_contour
		from optuna.visualization.matplotlib import plot_edf
		from optuna.visualization.matplotlib import plot_intermediate_values
		from optuna.visualization.matplotlib import plot_optimization_history
		from optuna.visualization.matplotlib import plot_parallel_coordinate
		from optuna.visualization.matplotlib import plot_param_importances
		from optuna.visualization.matplotlib import plot_slice

	SEED = 42
	np.random.seed(SEED)

	def objective(trial):
		data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
		train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
		dtrain = lgb.Dataset(train_x, label=train_y)
		dvalid = lgb.Dataset(valid_x, label=valid_y)

		param = {
			"objective": "binary",
			"metric": "auc",
			"verbosity": -1,
			"boosting_type": "gbdt",
			"bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
			"bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
			"min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
		}

		# Add a callback for pruning.
		pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
		gbm = lgb.train(param, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback])

		preds = gbm.predict(valid_x)
		pred_labels = np.rint(preds)
		accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
		return accuracy

	study = optuna.create_study(
		direction="maximize",
		sampler=optuna.samplers.TPESampler(seed=SEED),
		pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
	)
	study.optimize(objective, n_trials=100, timeout=600)

	#--------------------
	# Visualize the optimization history.
	plot_optimization_history(study)

	# Visualize the learning curves of the trials.
	plot_intermediate_values(study)

	# Visualize high-dimensional parameter relationships.
	plot_parallel_coordinate(study)
	# Select parameters to visualize.
	plot_parallel_coordinate(study, params=["bagging_freq", "bagging_fraction"])

	# Visualize hyperparameter relationships.
	plot_contour(study)
	# Select parameters to visualize.
	plot_contour(study, params=["bagging_freq", "bagging_fraction"])

	# Visualize individual hyperparameters as slice plot.
	plot_slice(study)
	# Select parameters to visualize.
	plot_slice(study, params=["bagging_freq", "bagging_fraction"])

	# Visualize parameter importances.
	plot_param_importances(study)
	# Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
	plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")

	# Visualize empirical distribution function.
	plot_edf(study)

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html
def save_and_resume_study_with_RDB_backend_tutorial():
	# New study.
	#	We can create a persistent study by calling create_study() function as follows.
	#	An SQLite file example.db is automatically initialized with a new study record.

	# Add stream handler of stdout to show the messages.
	optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
	study_name = "example-study"  # Unique identifier of the study.
	storage_name = "sqlite:///{}.db".format(study_name)

	study = optuna.create_study(study_name=study_name, storage=storage_name)

	def objective(trial):
		x = trial.suggest_float("x", -10, 10)
		return (x - 2)**2

	study.optimize(objective, n_trials=3)

	# Resume study.
	#	To resume a study, instantiate a Study object passing the study name example-study and the DB URL sqlite:///example-study.db.

	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
	study.optimize(objective, n_trials=3)

	# Experimental history.
	#	We can access histories of studies and trials via the Study class.

	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
	df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

	print(df)

	print("Best params: {}".format(study.best_params))
	print("Best value: {}".format(study.best_value))
	print("Best Trial: {}".format(study.best_trial))
	print("Trials: {}".format(study.trials))

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html
def multi_objective_optimization_tutorial():
	import torch, torchvision
	import thop

	DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	DIR = ".."
	BATCHSIZE = 128
	N_TRAIN_EXAMPLES = BATCHSIZE * 30
	N_VALID_EXAMPLES = BATCHSIZE * 10

	def define_model(trial):
		n_layers = trial.suggest_int("n_layers", 1, 3)
		layers = []

		in_features = 28 * 28
		for i in range(n_layers):
			out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
			layers.append(torch.nn.Linear(in_features, out_features))
			layers.append(torch.nn.ReLU())
			p = trial.suggest_float("dropout_{}".format(i), 0.2, 0.5)
			layers.append(torch.nn.Dropout(p))

			in_features = out_features

		layers.append(torch.nn.Linear(in_features, 10))
		layers.append(torch.nn.LogSoftmax(dim=1))

		return torch.nn.Sequential(*layers)

	# Defines training and evaluation.
	def train_model(model, optimizer, train_loader):
		model.train()
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
			optimizer.zero_grad()
			torch.nn.functional.nll_loss(model(data), target).backward()
			optimizer.step()

	def eval_model(model, valid_loader):
		model.eval()
		correct = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(valid_loader):
				data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
				pred = model(data).argmax(dim=1, keepdim=True)
				correct += pred.eq(target.view_as(pred)).sum().item()

		accuracy = correct / N_VALID_EXAMPLES

		flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
		return flops, accuracy

	# Define multi-objective objective function. Objectives are FLOPS and accuracy.
	def objective(trial):
		train_dataset = torchvision.datasets.FashionMNIST(
			DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
		)
		train_loader = torch.utils.data.DataLoader(
			torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
			batch_size=BATCHSIZE,
			shuffle=True,
		)

		val_dataset = torchvision.datasets.FashionMNIST(
			DIR, train=False, transform=torchvision.transforms.ToTensor()
		)
		val_loader = torch.utils.data.DataLoader(
			torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
			batch_size=BATCHSIZE,
			shuffle=True,
		)
		model = define_model(trial).to(DEVICE)

		optimizer = torch.optim.Adam(
			model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
		)

		for epoch in range(10):
			train_model(model, optimizer, train_loader)
		flops, accuracy = eval_model(model, val_loader)

	# Run multi-objective optimization.
	study = optuna.create_study(directions=["minimize", "maximize"])
	study.optimize(objective, n_trials=30, timeout=300)

	print("Number of finished trials: ", len(study.trials))

	# Check trials on Pareto front visually.
	optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])

	# Fetch the list of trials on the Pareto front with best_trials.
	print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

	trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
	print(f"Trial with highest accuracy: ")
	print(f"\tnumber: {trial_with_highest_accuracy.number}")
	print(f"\tparams: {trial_with_highest_accuracy.params}")
	print(f"\tvalues: {trial_with_highest_accuracy.values}")

	# Learn which hyperparameters are affecting the flops most with hyperparameter importance.
	optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="flops")

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/003_attributes.html
def user_attributes_tutorial():
	import sklearn.datasets
	import sklearn.model_selection
	import sklearn.svm

	# Adding user attributes to studies.
	#	A Study object provides set_user_attr() method to register a pair of key and value as an user-defined attribute.
	#	A key is supposed to be a str, and a value be any object serializable with json.dumps.

	study = optuna.create_study(storage="sqlite:///example.db")
	study.set_user_attr("contributors", ["Akiba", "Sano"])
	study.set_user_attr("dataset", "MNIST")

	print("study.user_attrs: {}.".format(study.user_attrs))  # {'contributors': ['Akiba', 'Sano'], 'dataset': 'MNIST'}

	# StudySummary object, which can be retrieved by get_all_study_summaries(), also contains user-defined attributes.
	study_summaries = optuna.get_all_study_summaries("sqlite:///example.db")
	print("study_summaries[0].user_attrs: {}.".format(study_summaries[0].user_attrs))  # {"contributors": ["Akiba", "Sano"], "dataset": "MNIST"}

	# Adding tser attributes to trials.
	def objective(trial):
		iris = sklearn.datasets.load_iris()
		x, y = iris.data, iris.target

		svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
		clf = sklearn.svm.SVC(C=svc_c)
		accuracy = sklearn.model_selection.cross_val_score(clf, x, y).mean()

		trial.set_user_attr("accuracy", accuracy)

		return 1.0 - accuracy  # Return error for minimization.

	study.optimize(objective, n_trials=1)

	print("study.trials[0].user_attrs: {}.".format(study.trials[0].user_attrs))

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/004_cli.html
def command_line_interface_tutorial():
	# Command				Description
	# ask					Create a new trial and suggest parameters.
	# best-trial			Show the best trial.
	# best-trials			Show a list of trials located at the Pareto front.
	# create-study			Create a new study.
	# delete-study			Delete a specified study.
	# storage upgrade		Upgrade the schema of a storage.
	# studies				Show a list of studies.
	# study optimize		Start optimization of a study.
	# study set-user-attr	Set a user attribute to a study.
	# tell					Finish a trial, which was created by the ask command.
	# trials				Show a list of trials.

	# In foo.py.
	def objective(trial):
		x = trial.suggest_float("x", -10, 10)
		return (x - 2)**2

	# Please note that foo.py only contains the definition of the objective function.
	# By giving the script file name and the method name of objective function to optuna study optimize command, we can invoke the optimization.
	#$ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
	#$ optuna study optimize foo.py objective --n-trials=100 --storage sqlite:///example.db --study-name $STUDY_NAME

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html
def user_defined_sampler_tutorial():
	raise NotImplementedError

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/006_user_defined_pruner.html
def user_defined_pruner_tutorial():
	raise NotImplementedError

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html
def callback_for_study_optimize_tutorial():
	# This example implements a stateful callback which stops the optimization if a certain number of trials are pruned in a row.
	# The number of trials pruned in a row is specified by threshold.
	class StopWhenTrialKeepBeingPrunedCallback:
		def __init__(self, threshold: int):
			self.threshold = threshold
			self._consequtive_pruned_count = 0

		def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
			if trial.state == optuna.trial.TrialState.PRUNED:
				self._consequtive_pruned_count += 1
			else:
				self._consequtive_pruned_count = 0

			if self._consequtive_pruned_count >= self.threshold:
				study.stop()

	# This objective prunes all the trials except for the first 5 trials (trial.number starts with 0).
	def objective(trial):
		if trial.number > 4:
			raise optuna.TrialPruned

		return trial.suggest_float("x", 0, 1)

	# Here, we set the threshold to 2: optimization finishes once two trials are pruned in a row.
	# So, we expect this study to stop after 7 trials.

	# Add stream handler of stdout to show the messages.
	optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

	study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(2)
	study = optuna.create_study()
	study.optimize(objective, n_trials=10, callbacks=[study_stop_cb])

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/008_specify_params.html
def specify_hyperparameters_manually_tutorial():
	raise NotImplementedError

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html
def ask_and_tell_interface_tutorial():
	raise NotImplementedError

# REF [site] >> https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/010_reuse_best_trial.html
def re_use_the_best_trial_tutorial():
	from sklearn import metrics
	from sklearn.datasets import make_classification
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import train_test_split

	def objective(trial):
		X, y = make_classification(n_features=10, random_state=1)
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

		C = trial.suggest_float("C", 1e-7, 10.0, log=True)

		clf = LogisticRegression(C=C)
		clf.fit(X_train, y_train)

		return clf.score(X_test, y_test)

	study = optuna.create_study(direction="maximize")
	study.optimize(objective, n_trials=10)

	print(study.best_trial.value)  # Show the best value.

	#--------------------
	# Suppose after the hyperparameter optimization, you want to calculate other evaluation metrics such as recall, precision, and f1-score on the same dataset.
	# You can define another objective function that shares most of the objective function to reproduce the model with the best hyperparameters.

	def detailed_objective(trial):
		# Use same code objective to reproduce the best model.
		X, y = make_classification(n_features=10, random_state=1)
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

		C = trial.suggest_float("C", 1e-7, 10.0, log=True)

		clf = LogisticRegression(C=C)
		clf.fit(X_train, y_train)

		# calculate more evaluation metrics
		pred = clf.predict(X_test)

		acc = metrics.accuracy_score(pred, y_test)
		recall = metrics.recall_score(pred, y_test)
		precision = metrics.precision_score(pred, y_test)
		f1 = metrics.f1_score(pred, y_test)

		return acc, f1, recall, precision

	# Pass study.best_trial as the argument of detailed_objective.
	detailed_objective(study.best_trial)  # Calculate acc, f1, recall, and precision

	# The difference between best_trial and ordinal trials.
	#	This uses best_trial, which returns the best_trial as a FrozenTrial.
	#	The FrozenTrial is different from an active trial and behaves differently from Trial in some situations.
	#	For example, pruning does not work because should_prune always returns False.

# REF [site] >> https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py
def xgboost_integration_example():
	import sklearn.datasets
	import sklearn.metrics
	from sklearn.model_selection import train_test_split
	import xgboost as xgb

	# FYI: Objective functions can take additional arguments.
	# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
	def objective(trial):
		data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
		train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
		dtrain = xgb.DMatrix(train_x, label=train_y)
		dvalid = xgb.DMatrix(valid_x, label=valid_y)

		param = {
			"verbosity": 0,
			"objective": "binary:logistic",
			"eval_metric": "auc",
			"booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
			"lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
			"alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
		}

		if param["booster"] == "gbtree" or param["booster"] == "dart":
			param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
			param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
			param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
			param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
		if param["booster"] == "dart":
			param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
			param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
			param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
			param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

		# Add a callback for pruning.
		pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
		bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback])
		preds = bst.predict(dvalid)
		pred_labels = np.rint(preds)
		accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
		return accuracy

	study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize")
	study.optimize(objective, n_trials=100)
	print(study.best_trial)

def main():
	# REF [site] >> https://github.com/optuna/optuna-examples/

	simple_quadratic_function_tutorial()
	#search_space_tutorial()
	#efficient_optimization_algorithms_tutorial()
	#easy_parallelization_tutorial()
	#quick_visualization_for_hyperparameter_optimization_analysis_tutorial()

	#save_and_resume_study_with_RDB_backend_tutorial()
	#multi_objective_optimization_tutorial()
	#user_attributes_tutorial()
	#command_line_interface_tutorial()
	#user_defined_sampler_tutorial()  # Not yet implemented.
	#user_defined_pruner_tutorial()  # Not yet implemented.
	#callback_for_study_optimize_tutorial()
	#specify_hyperparameters_manually_tutorial()  # Not yet implemented.
	#ask_and_tell_interface_tutorial()  # Not yet implemented.
	#re_use_the_best_trial_tutorial()

	#--------------------
	# XGBoost.
	#	REF [site] >> https://github.com/optuna/optuna-examples/tree/main/xgboost

	#xgboost_integration_example()

	# scikit-learn.
	#	REF [site] >> https://github.com/optuna/optuna-examples/tree/main/sklearn

	# PyTorch.
	#	REF [site] >> https://github.com/optuna/optuna-examples/tree/main/pytorch

	# TensorFlow.
	#	REF [site] >> https://github.com/optuna/optuna-examples/tree/main/tensorflow

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
