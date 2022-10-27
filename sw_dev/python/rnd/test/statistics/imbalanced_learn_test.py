#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from collections import Counter
import numpy as np
import sklearn.svm, sklearn.ensemble, sklearn.tree, sklearn.model_selection, sklearn.metrics, sklearn.datasets
import imblearn
import imblearn.over_sampling, imblearn.under_sampling, imblearn.combine
import imblearn.pipeline, imblearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns

def create_dataset(
	n_samples=1000,
	weights=(0.01, 0.01, 0.98),
	n_classes=3,
	class_sep=0.8,
	n_clusters=1,
):
	return sklearn.datasets.make_classification(
		n_samples=n_samples,
		n_features=2,
		n_informative=2,
		n_redundant=0,
		n_repeated=0,
		n_classes=n_classes,
		n_clusters_per_class=n_clusters,
		weights=list(weights),
		class_sep=class_sep,
		random_state=0,
	)

def plot_resampling(X, y, sampler, ax, title=None):
	X_res, y_res = sampler.fit_resample(X, y)
	ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
	if title is None:
		title = f"Resampling with {sampler.__class__.__name__}"
	ax.set_title(title)
	sns.despine(ax=ax, offset=10)

def plot_decision_function(X, y, clf, ax, title=None):
	plot_step = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))

	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	ax.contourf(xx, yy, Z, alpha=0.4)
	ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor="k")
	if title is not None:
		ax.set_title(title)

# REF [site] >> https://imbalanced-learn.org/stable/over_sampling.html
def over_sampling_example():
	X, y = sklearn.datasets.make_classification(
		n_samples=5000, n_features=2, n_informative=2,
		n_redundant=0, n_repeated=0, n_classes=3,
		n_clusters_per_class=1,
		weights=[0.01, 0.05, 0.94],
		class_sep=0.8, random_state=0
	)

	# Naive random over-sampling.
	# One way to fight this issue is to generate new samples in the classes which are under-represented.
	# The most naive strategy is to generate new samples by randomly sampling with replacement the current available samples.
	ros = imblearn.over_sampling.RandomOverSampler(random_state=0)
	X_resampled, y_resampled = ros.fit_resample(X, y)

	print(sorted(Counter(y_resampled).items()))

	clf = sklearn.svm.LinearSVC()
	clf.fit(X_resampled, y_resampled)

	X, y = create_dataset(n_samples=100, weights=(0.05, 0.25, 0.7))

	clf.fit(X, y)

	sampler = imblearn.over_sampling.RandomOverSampler(random_state=0)
	model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
	plot_decision_function(X, y, clf, axs[0], title="Without resampling")
	plot_decision_function(X, y, model, axs[1], f"Using {model[0].__class__.__name__}")
	fig.suptitle(f"Decision function of {clf.__class__.__name__}")
	fig.tight_layout()

	# RandomOverSampler allows to sample heterogeneous data.
	X_hetero = np.array([["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object)
	y_hetero = np.array([0, 0, 1])
	X_resampled, y_resampled = ros.fit_resample(X_hetero, y_hetero)

	print(X_resampled)
	print(y_resampled)

	# Work with pandas DataFrame.
	df_adult, y_adult = sklearn.datasets.fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
	print(df_adult.head())

	df_resampled, y_resampled = ros.fit_resample(df_adult, y_adult)
	print(df_resampled.head())

	fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
	sampler.set_params(shrinkage=None)
	plot_resampling(X, y, sampler, ax=axs[0], title="Normal bootstrap")
	sampler.set_params(shrinkage=0.3)
	plot_resampling(X, y, sampler, ax=axs[1], title="Smoothed bootstrap")
	fig.suptitle(f"Resampling with {sampler.__class__.__name__}")
	fig.tight_layout()

	#-----
	# From random over-sampling to SMOTE and ADASYN.
	# Apart from the random sampling with replacement, there are two popular methods to over-sample minority classes:
	#	(i) the Synthetic Minority Oversampling Technique (SMOTE).
	#	(ii) the Adaptive Synthetic (ADASYN) sampling method.
	#X_resampled, y_resampled = imblearn.over_sampling.SMOTE().fit_resample(X, y)
	X_resampled, y_resampled = imblearn.over_sampling.SMOTE(k_neighbors=3).fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	clf_smote = sklearn.svm.LinearSVC().fit(X_resampled, y_resampled)

	#X_resampled, y_resampled = imblearn.over_sampling.ADASYN().fit_resample(X, y)
	X_resampled, y_resampled = imblearn.over_sampling.ADASYN(n_neighbors=3).fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	clf_adasyn = sklearn.svm.LinearSVC().fit(X_resampled, y_resampled)

	X, y = create_dataset(n_samples=150, weights=(0.1, 0.2, 0.7))

	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
	samplers = [
		imblearn.FunctionSampler(),  # Identity resampler.
		imblearn.over_sampling.RandomOverSampler(random_state=0),
		imblearn.over_sampling.SMOTE(random_state=0),
		imblearn.over_sampling.ADASYN(random_state=0),
	]
	for ax, sampler in zip(axs.ravel(), samplers):
		title = "Original dataset" if isinstance(sampler, imblearn.FunctionSampler) else None
		plot_resampling(X, y, sampler, ax, title=title)
	fig.tight_layout()

	#-----
	# Ill-posed examples.
	# While the RandomOverSampler is over-sampling by duplicating some of the original samples of the minority class, SMOTE and ADASYN generate new samples in by interpolation.
	# However, the samples used to interpolate/generate new synthetic samples differ.
	# In fact, ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier while the basic implementation of SMOTE will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule.
	X, y = create_dataset(n_samples=150, weights=(0.05, 0.25, 0.7))

	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
	models = {
		"Without sampler": clf,
		"ADASYN sampler": imblearn.pipeline.make_pipeline(imblearn.over_sampling.ADASYN(random_state=0), clf),
		"SMOTE sampler": imblearn.pipeline.make_pipeline(imblearn.over_sampling.SMOTE(random_state=0), clf),
	}
	for ax, (title, model) in zip(axs, models.items()):
		model.fit(X, y)
		plot_decision_function(X, y, model, ax=ax, title=title)
	fig.suptitle(f"Decision function using a {clf.__class__.__name__}")
	fig.tight_layout()

	# The sampling particularities of these two algorithms can lead to some peculiar behavior as shown below.
	X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)

	samplers = [imblearn.over_sampling.SMOTE(random_state=0), imblearn.over_sampling.ADASYN(random_state=0)]
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, clf, ax[0], title=f"Decision function with {sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1])
	fig.suptitle("Particularities of over-sampling with SMOTE and ADASYN")
	fig.tight_layout()

	#-----
	# SMOTE variants.
	# SMOTE might connect inliers and outliers while ADASYN might focus solely on outliers which, in both cases, might lead to a sub-optimal decision function.
	# In this regard, SMOTE offers three additional options to generate samples.
	# Those methods focus on samples near the border of the optimal decision function and will generate samples in the opposite direction of the nearest neighbors class.
	X, y = create_dataset(n_samples=5000, weights=(0.01, 0.05, 0.94), class_sep=0.8)

	fig, axs = plt.subplots(5, 2, figsize=(15, 30))
	samplers = [
		imblearn.over_sampling.SMOTE(random_state=0),
		imblearn.over_sampling.BorderlineSMOTE(random_state=0, kind="borderline-1"),
		imblearn.over_sampling.BorderlineSMOTE(random_state=0, kind="borderline-2"),
		imblearn.over_sampling.KMeansSMOTE(random_state=0),
		imblearn.over_sampling.SVMSMOTE(random_state=0),
	]
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, clf, ax[0], title=f"Decision function for {sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1])
	fig.suptitle("Decision function and resampling using SMOTE variants")
	fig.tight_layout()

	# The BorderlineSMOTE, SVMSMOTE, and KMeansSMOTE offer some variant of the SMOTE algorithm.
	X_resampled, y_resampled = imblearn.over_sampling.BorderlineSMOTE().fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# When dealing with mixed data type such as continuous and categorical features, none of the presented methods (apart of the class RandomOverSampler) can deal with the categorical features.
	# The SMOTENC is an extension of the SMOTE algorithm for which categorical data are treated differently.
	# Create a synthetic data set with continuous and categorical features.
	rng = np.random.RandomState(42)
	n_samples = 50
	X = np.empty((n_samples, 3), dtype=object)
	X[:, 0] = rng.choice(["A", "B", "C"], size=n_samples).astype(object)
	X[:, 1] = rng.randn(n_samples)
	X[:, 2] = rng.randint(3, size=n_samples)
	y = np.array([0] * 20 + [1] * 30)
	print(sorted(Counter(y).items()))

	# In this data set, the first and last features are considered as categorical features.
	# One need to provide this information to SMOTENC via the parameters categorical_features either by passing the indices of these features or a boolean mask marking these features.
	smote_nc = imblearn.over_sampling.SMOTENC(categorical_features=[0, 2], random_state=0)
	X_resampled, y_resampled = smote_nc.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))
	print(X_resampled[-5:])

	# However, SMOTENC is only working when data is a mixed of numerical and categorical features.
	# If data are made of only categorical data, one can use the SMOTEN variant.
	# The algorithm changes in two ways:
	#	The nearest neighbors search does not rely on the Euclidean distance. Indeed, the value difference metric (VDM) also implemented in the class ValueDifferenceMetric is used.
	#	A new sample is generated where each feature value corresponds to the most common category seen in the neighbors samples belonging to the same class.
	X = np.array(["green"] * 5 + ["red"] * 10 + ["blue"] * 7, dtype=object).reshape(-1, 1)
	y = np.array(["apple"] * 5 + ["not apple"] * 3 + ["apple"] * 7 + ["not apple"] * 5 + ["apple"] * 2, dtype=object)

	sampler = imblearn.over_sampling.SMOTEN(random_state=0)
	X_res, y_res = sampler.fit_resample(X, y)

	print(X_res[y.size:])
	print(y_res[y.size:])

	plt.show()

# REF [site] >> https://imbalanced-learn.org/stable/under_sampling.html
def under_sampling_example():
	X, y = sklearn.datasets.make_classification(
		n_samples=5000, n_features=2, n_informative=2,
		n_redundant=0, n_repeated=0, n_classes=3,
		n_clusters_per_class=1,
		weights=[0.01, 0.05, 0.94],
		class_sep=0.8, random_state=0
	)
	print(sorted(Counter(y).items()))

	clf = sklearn.svm.LinearSVC()

	# Prototype generation.
	cc = imblearn.under_sampling.ClusterCentroids(random_state=0)
	X_resampled, y_resampled = cc.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	X, y = create_dataset(n_samples=400, weights=(0.05, 0.15, 0.8), class_sep=0.8)

	samplers = {
		imblearn.FunctionSampler(),  # Identity resampler.
		imblearn.under_sampling.ClusterCentroids(random_state=0),
	}
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, model, ax[0], title=f"Decision function with {sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1])
	fig.tight_layout()

	#-----
	# Prototype selection.

	# Controlled under-sampling techniques.
	rus = imblearn.under_sampling.RandomUnderSampler(random_state=0)
	X_resampled, y_resampled = rus.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	X, y = create_dataset(n_samples=400, weights=(0.05, 0.15, 0.8), class_sep=0.8)

	samplers = {
		imblearn.FunctionSampler(),  # Identity resampler.
		imblearn.under_sampling.RandomUnderSampler(random_state=0),
	}
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, model, ax[0], title=f"Decision function with {sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1])
	fig.tight_layout()

	# RandomUnderSampler allows to bootstrap the data by setting replacement to True.
	# The resampling with multiple classes is performed by considering independently each targeted class.
	print(np.vstack([tuple(row) for row in X_resampled]).shape)

	rus = imblearn.under_sampling.RandomUnderSampler(random_state=0, replacement=True)
	X_resampled, y_resampled = rus.fit_resample(X, y)
	print(np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape)

	# RandomUnderSampler allows to sample heterogeneous data.
	X_hetero = np.array([["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=object)
	y_hetero = np.array([0, 0, 1])
	X_resampled, y_resampled = rus.fit_resample(X_hetero, y_hetero)

	print(X_resampled)
	print(y_resampled)

	# Work with pandas DataFrame.
	df_adult, y_adult = sklearn.datasets.fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
	print(df_adult.head())

	df_resampled, y_resampled = rus.fit_resample(df_adult, y_adult)
	print(df_resampled.head())

	# NearMiss adds some heuristic rules to select samples.
	nm1 = imblearn.under_sampling.NearMiss(version=1)
	X_resampled_nm1, y_resampled = nm1.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	X, y = create_dataset(n_samples=1000, weights=(0.05, 0.15, 0.8), class_sep=1.5)

	samplers = [imblearn.under_sampling.NearMiss(version=1), imblearn.under_sampling.NearMiss(version=2), imblearn.under_sampling.NearMiss(version=3)]
	fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 25))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, model, ax[0], title=f"Decision function for {sampler.__class__.__name__}-{sampler.version}")
		plot_resampling(X, y, sampler, ax[1], title=f"Resampling using {sampler.__class__.__name__}-{sampler.version}")
	fig.tight_layout()

	# Edited data set using nearest neighbours.
	# EditedNearestNeighbours applies a nearest-neighbors algorithm and "edit" the dataset by removing samples which do not agree "enough" with their neighboorhood.
	# For each sample in the class to be under-sampled, the nearest-neighbours are computed and if the selection criterion is not fulfilled, the sample is removed.
	print(sorted(Counter(y).items()))

	enn = imblearn.under_sampling.EditedNearestNeighbours()
	X_resampled, y_resampled = enn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# Two selection criteria are currently available: (i) the majority (i.e., kind_sel='mode') or (ii) all (i.e., kind_sel='all') the nearest-neighbors have to belong to the same class than the sample inspected to keep it in the dataset.
	# Thus, it implies that kind_sel='all' will be less conservative than kind_sel='mode', and more samples will be excluded in the former strategy than the latest.
	enn = imblearn.under_sampling.EditedNearestNeighbours(kind_sel="all")
	X_resampled, y_resampled = enn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	enn = imblearn.under_sampling.EditedNearestNeighbours(kind_sel="mode")
	X_resampled, y_resampled = enn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# The parameter n_neighbors allows to give a classifier subclassed from KNeighborsMixin from scikit-learn to find the nearest neighbors and make the decision to keep a given sample or not.

	# RepeatedEditedNearestNeighbours extends EditedNearestNeighbours by repeating the algorithm multiple times.
	# Generally, repeating the algorithm will delete more data.
	renn = imblearn.under_sampling.RepeatedEditedNearestNeighbours()
	X_resampled, y_resampled = renn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# AllKNN differs from the previous RepeatedEditedNearestNeighbours since the number of neighbors of the internal nearest neighbors algorithm is increased at each iteration.
	allknn = imblearn.under_sampling.AllKNN()
	X_resampled, y_resampled = allknn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	from imblearn.under_sampling import (
		EditedNearestNeighbours,
		RepeatedEditedNearestNeighbours,
		AllKNN,
	)

	X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

	samplers = [
		imblearn.under_sampling.EditedNearestNeighbours(),
		imblearn.under_sampling.RepeatedEditedNearestNeighbours(),
		imblearn.under_sampling.AllKNN(allow_minority=True),
	]
	fig, axs = plt.subplots(3, 2, figsize=(15, 25))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		#plot_decision_function(X, y, clf, ax[0], title=f"Decision function for \n{sampler.__class__.__name__}")
		plot_decision_function(X, y, model, ax[0], title=f"Decision function for \n{sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1], title=f"Resampling using \n{sampler.__class__.__name__}")
	fig.tight_layout()

	# Condensed nearest neighbors and derived algorithms.
	# CondensedNearestNeighbour uses a 1 nearest neighbor rule to iteratively decide if a sample should be removed or not.
	cnn = imblearn.under_sampling.CondensedNearestNeighbour(random_state=0)
	X_resampled, y_resampled = cnn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# In the contrary, OneSidedSelection will use TomekLinks to remove noisy samples.
	# In addition, the 1 nearest neighbor rule is applied to all samples and the one which are misclassified will be added to the set C.
	# No iteration on the set S will take place.
	oss = imblearn.under_sampling.OneSidedSelection(random_state=0)
	X_resampled, y_resampled = oss.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	# NeighbourhoodCleaningRule will focus on cleaning the data than condensing them.
	# Therefore, it will used the union of samples to be rejected between the EditedNearestNeighbours and the output a 3 nearest neighbors classifier.
	ncr = imblearn.under_sampling.NeighbourhoodCleaningRule()
	X_resampled, y_resampled = ncr.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	X, y = create_dataset(n_samples=500, weights=(0.2, 0.3, 0.5), class_sep=0.8)

	fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 25))
	samplers = [
		imblearn.under_sampling.CondensedNearestNeighbour(random_state=0),
		imblearn.under_sampling.OneSidedSelection(random_state=0),
		imblearn.under_sampling.NeighbourhoodCleaningRule(),
	]
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		#plot_decision_function(X, y, clf, ax[0], title=f"Decision function for \n{sampler.__class__.__name__}")
		plot_decision_function(X, y, model, ax[0], title=f"Decision function for \n{sampler.__class__.__name__}")
		plot_resampling(X, y, sampler, ax[1], title=f"Resampling using \n{sampler.__class__.__name__}")
	fig.tight_layout()

	# Instance hardness threshold.
	# InstanceHardnessThreshold is a specific algorithm in which a classifier is trained on the data and the samples with lower probabilities are removed.
	iht = imblearn.under_sampling.InstanceHardnessThreshold(
		random_state=0,
		estimator=sklearn.linear_model.LogisticRegression(solver="lbfgs", multi_class="auto"),
	)
	X_resampled, y_resampled = iht.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	samplers = {
		imblearn.FunctionSampler(),  # Identity resampler.
		imblearn.under_sampling.InstanceHardnessThreshold(estimator=sklearn.linear_model.LogisticRegression(), random_state=0),
	}
	fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
	for ax, sampler in zip(axs, samplers):
		model = imblearn.pipeline.make_pipeline(sampler, clf).fit(X, y)
		plot_decision_function(X, y, model, ax[0], title=f"Decision function with \n{sampler.__class__.__name__}",)
		plot_resampling(X, y, sampler, ax[1], title=f"Resampling using \n{sampler.__class__.__name__}")
	fig.tight_layout()

	plt.show()

# REF [site] >> https://imbalanced-learn.org/stable/combine.html
def combination_of_over_and_under_sampling_example():
	X, y = sklearn.datasets.make_classification(
		n_samples=5000, n_features=2, n_informative=2,
		n_redundant=0, n_repeated=0, n_classes=3,
		n_clusters_per_class=1,
		weights=[0.01, 0.05, 0.94],
		class_sep=0.8, random_state=0,
	)
	print(sorted(Counter(y).items()))

	smote_enn = imblearn.combine.SMOTEENN(random_state=0)
	X_resampled, y_resampled = smote_enn.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	smote_tomek = imblearn.combine.SMOTETomek(random_state=0)
	X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
	print(sorted(Counter(y_resampled).items()))

	X, y = sklearn.datasets.make_classification(
		n_samples=100,
		n_features=2,
		n_informative=2,
		n_redundant=0,
		n_repeated=0,
		n_classes=3,
		n_clusters_per_class=1,
		weights=[0.1, 0.2, 0.7],
		class_sep=0.8,
		random_state=0,
	)

	def plot_resampling(X, y, sampler, ax):
		"""Plot the resampled dataset using the sampler."""
		X_res, y_res = sampler.fit_resample(X, y)
		ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
		sns.despine(ax=ax, offset=10)
		ax.set_title(f"Decision function for {sampler.__class__.__name__}")
		return Counter(y_res)

	samplers = [imblearn.over_sampling.SMOTE(random_state=0), imblearn.combine.SMOTEENN(random_state=0), imblearn.combine.SMOTETomek(random_state=0)]
	fig, axs = plt.subplots(3, 2, figsize=(15, 25))
	for ax, sampler in zip(axs, samplers):
		clf = imblearn.pipeline.make_pipeline(sampler, sklearn.svm.LinearSVC()).fit(X, y)
		plot_decision_function(X, y, clf, ax[0])
		plot_resampling(X, y, sampler, ax[1])
	fig.tight_layout()

	plt.show()

# REF [site] >> https://imbalanced-learn.org/stable/ensemble.html
def ensemble_of_samplers():
	# Classifier including inner balancing samplers.
	X, y = sklearn.datasets.make_classification(
		n_samples=10000, n_features=2, n_informative=2,
		n_redundant=0, n_repeated=0, n_classes=3,
		n_clusters_per_class=1,
		weights=[0.01, 0.05, 0.94], class_sep=0.8,
		random_state=0,
	)

	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=0)

	# Bagging classifier.
	# In ensemble classifiers, bagging methods build several estimators on different randomly selected subset of data.
	# In scikit-learn, this classifier is named BaggingClassifier.
	# However, this classifier does not allow to balance each subset of data.
	# Therefore, when training on imbalanced data set, this classifier will favor the majority classes.
	bc = sklearn.ensemble.BaggingClassifier(base_estimator=sklearn.tree.DecisionTreeClassifier(), random_state=0)
	bc.fit(X_train, y_train)

	y_pred = bc.predict(X_test)
	print("balanced_accuracy_score = {}.".format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

	# In BalancedBaggingClassifier, each bootstrap sample will be further resampled to achieve the sampling_strategy desired.
	# Therefore, BalancedBaggingClassifier takes the same parameters as the scikit-learn BaggingClassifier.
	# In addition, the sampling is controlled by the parameter sampler or the two parameters sampling_strategy and replacement, if one wants to use the RandomUnderSampler:
	bbc = imblearn.ensemble.BalancedBaggingClassifier(
		base_estimator=sklearn.tree.DecisionTreeClassifier(),
		sampling_strategy="auto",
		replacement=False,
		random_state=0,
	)
	bbc.fit(X_train, y_train)

	y_pred = bbc.predict(X_test)
	print("balanced_accuracy_score = {}.".format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

	# Forest of randomized trees.
	# BalancedRandomForestClassifier is another ensemble method in which each tree of the forest will be provided a balanced bootstrap sample.
	brf = imblearn.ensemble.BalancedRandomForestClassifier(n_estimators=100, random_state=0)
	brf.fit(X_train, y_train)

	y_pred = brf.predict(X_test)
	print("balanced_accuracy_score = {}.".format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

	# Boosting.
	# RUSBoostClassifier randomly under-sample the dataset before to perform a boosting iteration.
	rusboost = imblearn.ensemble.RUSBoostClassifier(n_estimators=200, algorithm="SAMME.R", random_state=0)
	rusboost.fit(X_train, y_train)

	y_pred = rusboost.predict(X_test)
	print("balanced_accuracy_score = {}.".format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

	# A specific method which uses AdaBoostClassifier as learners in the bagging classifier is called "EasyEnsemble".
	# The EasyEnsembleClassifier allows to bag AdaBoost learners which are trained on balanced bootstrap samples.
	eec = imblearn.ensemble.EasyEnsembleClassifier(random_state=0)
	eec.fit(X_train, y_train)

	y_pred = eec.predict(X_test)
	print("balanced_accuracy_score = {}.".format(sklearn.metrics.balanced_accuracy_score(y_test, y_pred)))

# REF [site] >> https://imbalanced-learn.org/stable/miscellaneous.html
def miscellaneous_samplers_example():
	raise NotImplementedError

# REF [site] >> https://imbalanced-learn.org/stable/metrics.html
def metrics_example():
	raise NotImplementedError

# REF [site] >> https://imbalanced-learn.org/stable/common_pitfalls.html
def common_pitfalls_and_recommended_practices_example():
	# Data leakage.
	# Data leakage occurs when information that would not be available at prediction time is used when building the model.

	# In the resampling setting, there is a common pitfall that corresponds to resample the entire dataset before splitting it into a train and a test partitions.
	# Note that it would be equivalent to resample the train and test partitions as well.
	# Such of a processing leads to two issues:
	#	The model will not be tested on a dataset with class distribution similar to the real use-case.
	#		Indeed, by resampling the entire dataset, both the training and testing set will be potentially balanced while the model should be tested on the natural imbalanced dataset to evaluate the potential bias of the model.
	#	The resampling procedure might use information about samples in the dataset to either generate or select some of the samples.
	#		Therefore, we might use information of samples which will be later used as testing samples which is the typical data leakage issue.

	# For the sake of simplicity, we will only use the numerical features.
	# Also, we will make the dataset more imbalanced to increase the effect of the wrongdoings.
	X, y = sklearn.datasets.fetch_openml(data_id=1119, as_frame=True, return_X_y=True)
	X = X.select_dtypes(include="number")
	X, y = imblearn.datasets.make_imbalance(X, y, sampling_strategy={">50K": 300}, random_state=1)

	print(y.value_counts(normalize=True))

	# To later highlight some of the issue, we will keep aside a left-out set that we will not use for the evaluation of the model.
	X, X_left_out, y, y_left_out = sklearn.model_selection.train_test_split(X, y, stratify=y, random_state=0)

	# We will train and check the performance of this classifier, without any preprocessing to alleviate the bias toward the majority class.
	# We evaluate the generalization performance of the classifier via cross-validation.
	model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
	cv_results = sklearn.model_selection.cross_validate(
		model, X, y, scoring="balanced_accuracy",
		return_train_score=True, return_estimator=True,
		n_jobs=-1,
	)
	print(
		f"Balanced accuracy mean +/- std. dev.: "
		f"{cv_results["test_score"].mean():.3f} +/- "
		f"{cv_results["test_score"].std():.3f}"
	)

	scores = []
	for fold_id, cv_model in enumerate(cv_results["estimator"]):
		scores.append(sklearn.metrics.balanced_accuracy_score(y_left_out, cv_model.predict(X_left_out)))
	print(
		f"Balanced accuracy mean +/- std. dev.: "
		f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
	)

	# Let's now show the wrong pattern to apply when it comes to resampling to alleviate the class imbalance issue.
	# We will use a sampler to balance the entire dataset and check the statistical performance of our classifier via cross-validation.
	sampler = imblearn.under_sampling.RandomUnderSampler(random_state=0)
	X_resampled, y_resampled = sampler.fit_resample(X, y)
	model = sklearn.ensemble.HistGradientBoostingClassifier(random_state=0)
	cv_results = sklearn.model_selection.cross_validate(
		model, X_resampled, y_resampled, scoring="balanced_accuracy",
		return_train_score=True, return_estimator=True,
		n_jobs=-1,
	)
	print(
		f"Balanced accuracy mean +/- std. dev.: "
		f"{cv_results["test_score"].mean():.3f} +/- "
		f"{cv_results["test_score"].std():.3f}"
	)

	# We will now illustrate the correct pattern to use.
	# Indeed, as in scikit-learn, using a Pipeline avoids to make any data leakage because the resampling will be delegated to imbalanced-learn and does not require any manual steps.
	model = imblearn.pipeline.make_pipeline(
		imblearn.under_sampling.RandomUnderSampler(random_state=0),
		sklearn.ensemble.HistGradientBoostingClassifier(random_state=0),
	)
	cv_results = sklearn.model_selection.cross_validate(
		model, X, y, scoring="balanced_accuracy",
		return_train_score=True, return_estimator=True,
		n_jobs=-1,
	)
	print(
		f"Balanced accuracy mean +/- std. dev.: "
		f"{cv_results["test_score"].mean():.3f} +/- "
		f"{cv_results["test_score"].std():.3f}"
	)

	# We see that the statistical performance are very close to the cross-validation study that we perform, without any sign of over-optimistic results.
	scores = []
	for fold_id, cv_model in enumerate(cv_results["estimator"]):
		scores.append(sklearn.metrics.balanced_accuracy_score(y_left_out, cv_model.predict(X_left_out)))
	print(
		f"Balanced accuracy mean +/- std. dev.: "
		f"{np.mean(scores):.3f} +/- {np.std(scores):.3f}"
	)

# REF [site] >> https://imbalanced-learn.org/stable/datasets/index.html
def dataset_loading_utilities_example():
	raise NotImplementedError

def main():
	over_sampling_example()
	#under_sampling_example()
	#combination_of_over_and_under_sampling_example()
	#ensemble_of_samplers()
	#miscellaneous_samplers_example()  # Not yet implemented.
	#metrics_example()  # Not yet implemented.
	#common_pitfalls_and_recommended_practices_example()
	#dataset_loading_utilities_example()  # Not yet implemented.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
