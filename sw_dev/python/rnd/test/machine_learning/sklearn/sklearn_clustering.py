#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://scikit-learn.org/stable/modules/clustering.html
#	https://docs.scipy.org/doc/scipy-0.14.0/reference/cluster.hierarchy.html
#	http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#	http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#
#	Dendrogram:
#		https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

import warnings, itertools, time
from itertools import cycle, islice
import numpy as np
import sklearn.cluster, sklearn.datasets, sklearn.mixture, sklearn.neighbors, sklearn.preprocessing
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
def comparing_different_clustering_algorithms_on_toy_datasets_example():
	np.random.seed(0)

	# ============
	# Generate datasets. We choose the size big enough to see the scalability
	# of the algorithms, but not too big to avoid too long running times.
	# ============
	n_samples = 1500
	noisy_circles = sklearn.datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
	noisy_moons = sklearn.datasets.make_moons(n_samples=n_samples, noise=.05)
	blobs = sklearn.datasets.make_blobs(n_samples=n_samples, random_state=8)
	no_structure = np.random.rand(n_samples, 2), None

	# Anisotropicly distributed data.
	random_state = 170
	X, y = sklearn.datasets.make_blobs(n_samples=n_samples, random_state=random_state)
	transformation = [[0.6, -0.6], [-0.4, 0.8]]
	X_aniso = np.dot(X, transformation)
	aniso = (X_aniso, y)

	# Blobs with varied variances.
	varied = sklearn.datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

	# ============
	# Set up cluster parameters.
	# ============
	plt.figure(figsize=(9 * 2 + 3, 13))
	plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,hspace=.01)

	plot_num = 1

	default_base = {'quantile': .3,
					'eps': .3,
					'damping': .9,
					'preference': -200,
					'n_neighbors': 10,
					'n_clusters': 3,
					'min_samples': 20,
					'xi': 0.05,
					'min_cluster_size': 0.1}

	datasets = [
		(noisy_circles, {'damping': .77, 'preference': -240,
						 'quantile': .2, 'n_clusters': 2,
						 'min_samples': 20, 'xi': 0.25}),
		(noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
		(varied, {'eps': .18, 'n_neighbors': 2,
				  'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}),
		(aniso, {'eps': .15, 'n_neighbors': 2,
				 'min_samples': 20, 'xi': 0.1, 'min_cluster_size': .2}),
		(blobs, {}),
		(no_structure, {})]

	for i_dataset, (dataset, algo_params) in enumerate(datasets):
		# Update parameters with dataset-specific values.
		params = default_base.copy()
		params.update(algo_params)

		X, y = dataset

		# Normalize dataset for easier parameter selection.
		X = sklearn.preprocessing.StandardScaler().fit_transform(X)

		# Estimate bandwidth for mean shift.
		bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=params['quantile'])

		# Connectivity matrix for structured Ward.
		connectivity = sklearn.neighbors.kneighbors_graph(
			X, n_neighbors=params['n_neighbors'], include_self=False)
		# Make connectivity symmetric.
		connectivity = 0.5 * (connectivity + connectivity.T)

		# ============
		# Create cluster objects.
		# ============
		ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
		two_means = sklearn.cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
		ward = sklearn.cluster.AgglomerativeClustering(
			n_clusters=params['n_clusters'], linkage='ward',
			connectivity=connectivity)
		spectral = sklearn.cluster.SpectralClustering(
			n_clusters=params['n_clusters'], eigen_solver='arpack',
			affinity="nearest_neighbors")
		dbscan = sklearn.cluster.DBSCAN(eps=params['eps'])
		optics = sklearn.cluster.OPTICS(min_samples=params['min_samples'],
			xi=params['xi'], min_cluster_size=params['min_cluster_size'])
		affinity_propagation = sklearn.cluster.AffinityPropagation(
			damping=params['damping'], preference=params['preference'])
		average_linkage = sklearn.cluster.AgglomerativeClustering(
			linkage="average", affinity="cityblock",
			n_clusters=params['n_clusters'], connectivity=connectivity)
		birch = sklearn.cluster.Birch(n_clusters=params['n_clusters'])
		gmm = sklearn.mixture.GaussianMixture(
			n_components=params['n_clusters'], covariance_type='full')

		clustering_algorithms = (
			('MiniBatch\nKMeans', two_means),
			('Affinity\nPropagation', affinity_propagation),
			('MeanShift', ms),
			('Spectral\nClustering', spectral),
			('Ward', ward),
			('Agglomerative\nClustering', average_linkage),
			('DBSCAN', dbscan),
			('OPTICS', optics),
			('BIRCH', birch),
			('Gaussian\nMixture', gmm)
		)

		for name, algorithm in clustering_algorithms:
			t0 = time.time()

			# Catch warnings related to kneighbors_graph.
			with warnings.catch_warnings():
				warnings.filterwarnings(
					"ignore",
					message="the number of connected components of the " +
					"connectivity matrix is [0-9]{1,2}" +
					" > 1. Completing it to avoid stopping the tree early.",
					category=UserWarning)
				warnings.filterwarnings(
					"ignore",
					message="Graph is not fully connected, spectral embedding" +
					" may not work as expected.",
					category=UserWarning)
				algorithm.fit(X)

			t1 = time.time()
			if hasattr(algorithm, 'labels_'):
				y_pred = algorithm.labels_.astype(int)
			else:
				y_pred = algorithm.predict(X)

			plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
			if i_dataset == 0:
				plt.title(name, size=18)

			colors = np.array(list(itertools.islice(itertools.cycle(['#377eb8', '#ff7f00', '#4daf4a',
																	 '#f781bf', '#a65628', '#984ea3',
																	 '#999999', '#e41a1c', '#dede00']),
										  int(max(y_pred) + 1))))
			# Add black color for outliers (if any).
			colors = np.append(colors, ["#000000"])
			plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

			plt.xlim(-2.5, 2.5)
			plt.ylim(-2.5, 2.5)
			plt.xticks(())
			plt.yticks(())
			plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
					 transform=plt.gca().transAxes, size=15,
					 horizontalalignment='right')
			plot_num += 1

	plt.show()

def main():
	comparing_different_clustering_algorithms_on_toy_datasets_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
