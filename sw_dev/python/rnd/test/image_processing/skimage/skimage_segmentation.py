#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy
import skimage
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel
from skimage.segmentation import active_contour
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.feature import peak_local_max
from skimage.future import graph
from skimage.util import img_as_float
import matplotlib.pyplot as plt

# REF [site] >> https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
def watershed_example():
	# Generate an initial image with two overlapping circles.
	x, y = np.indices((80, 80))
	x1, y1, x2, y2 = 28, 28, 44, 52
	r1, r2 = 16, 20
	mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
	mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
	image = np.logical_or(mask_circle1, mask_circle2)

	# Now we want to separate the two objects in image.
	# Generate the markers as local maxima of the distance to the background.
	distance = scipy.ndimage.distance_transform_edt(image)
	local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
	markers = scipy.ndimage.label(local_maxi)[0]
	labels = watershed(-distance, markers, mask=image)

	fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
	ax = axes.ravel()

	ax[0].imshow(image, cmap=plt.cm.gray)
	ax[0].set_title('Overlapping objects')
	ax[1].imshow(-distance, cmap=plt.cm.gray)
	ax[1].set_title('Distances')
	ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
	ax[2].set_title('Separated objects')

	for a in ax:
		a.set_axis_off()

	fig.tight_layout()
	plt.show()

# REF [site] >> https://scikit-image.org/docs/stable/auto_examples/edges/plot_active_contours.html
def active_contour_example():
	img = data.astronaut()
	img = rgb2gray(img)

	s = np.linspace(0, 2 * np.pi, 400)
	x = 220 + 100 * np.cos(s)
	y = 100 + 100 * np.sin(s)
	init = np.array([x, y]).T

	snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10, gamma=0.001)

	fig, ax = plt.subplots(figsize=(7, 7))
	ax.imshow(img, cmap=plt.cm.gray)
	ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
	ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
	ax.set_xticks([]), ax.set_yticks([])
	ax.axis([0, img.shape[1], img.shape[0], 0])

	#--------------------
	img = data.text()

	x = np.linspace(5, 424, 100)
	y = np.linspace(136, 50, 100)
	init = np.array([x, y]).T

	snake = active_contour(gaussian(img, 1), init, bc='fixed', alpha=0.1, beta=1.0, w_line=-5, w_edge=0, gamma=0.1)

	fig, ax = plt.subplots(figsize=(9, 5))
	ax.imshow(img, cmap=plt.cm.gray)
	ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
	ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
	ax.set_xticks([]), ax.set_yticks([])
	ax.axis([0, img.shape[1], img.shape[0], 0])

	plt.show()

# REF [site] >> https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
def comparison_of_segmentation_and_superpixel_algorithms_example():
	img = img_as_float(data.astronaut()[::2, ::2])

	segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
	segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
	segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
	gradient = sobel(rgb2gray(img))
	segments_watershed = watershed(gradient, markers=250, compactness=0.001)

	print('Felzenszwalb number of segments: {}'.format(len(np.unique(segments_fz))))
	print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
	print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))

	fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

	ax[0, 0].imshow(mark_boundaries(img, segments_fz))
	ax[0, 0].set_title("Felzenszwalbs's method")
	ax[0, 1].imshow(mark_boundaries(img, segments_slic))
	ax[0, 1].set_title('SLIC')
	ax[1, 0].imshow(mark_boundaries(img, segments_quick))
	ax[1, 0].set_title('Quickshift')
	ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
	ax[1, 1].set_title('Compact watershed')

	for a in ax.ravel():
		a.set_axis_off()

	plt.tight_layout()
	plt.show()

def _weight_mean_color(graph, src, dst, n):
	"""Callback to handle merging nodes by recomputing mean color.

	The method expects that the mean color of 'dst' is already computed.

	Parameters
	----------
	graph : RAG
		The graph under consideration.
	src, dst : int
		The vertices in 'graph' to be merged.
	n : int
		A neighbor of 'src' or 'dst' or both.

	Returns
	-------
	data : dict
		A dictionary with the '"weight"' attribute set as the absolute
		difference of the mean color between node `dst` and `n`.
	"""

	diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
	diff = np.linalg.norm(diff)
	return {'weight': diff}

def merge_mean_color(graph, src, dst):
	"""Callback called before merging two nodes of a mean color distance graph.

	This method computes the mean color of 'dst'.

	Parameters
	----------
	graph : RAG
		The graph under consideration.
	src, dst : int
		The vertices in 'graph' to be merged.
	"""

	graph.node[dst]['total color'] += graph.node[src]['total color']
	graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
	graph.node[dst]['mean color'] = (graph.node[dst]['total color'] / graph.node[dst]['pixel count'])

# Region adjacency graph (RAG).
# REF [site] >> https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html
def rag_merging_example():
	img = data.coffee()

	labels = slic(img, compactness=30, n_segments=400)
	g = graph.rag_mean_color(img, labels)

	labels2 = graph.merge_hierarchical(
		labels, g, thresh=35, rag_copy=False,
		in_place_merge=True,
		merge_func=merge_mean_color,
		weight_func=_weight_mean_color
	)

	out = skimage.color.label2rgb(labels2, img, kind='avg')
	out = mark_boundaries(out, labels2, (0, 0, 0))
	skimage.io.imshow(out)
	skimage.io.show()

def weight_boundary(graph, src, dst, n):
	"""
	Handle merging of nodes of a region boundary region adjacency graph.

	This function computes the `"weight"` and the count `"count"`
	attributes of the edge between `n` and the node formed after
	merging `src` and `dst`.


	Parameters
	----------
	graph : RAG
		The graph under consideration.
	src, dst : int
		The vertices in `graph` to be merged.
	n : int
		A neighbor of `src` or `dst` or both.

	Returns
	-------
	data : dict
		A dictionary with the "weight" and "count" attributes to be
		assigned for the merged node.
	"""

	default = {'weight': 0.0, 'count': 0}

	count_src = graph[src].get(n, default)['count']
	count_dst = graph[dst].get(n, default)['count']

	weight_src = graph[src].get(n, default)['weight']
	weight_dst = graph[dst].get(n, default)['weight']

	count = count_src + count_dst
	return {
		'count': count,
		'weight': (count_src * weight_src + count_dst * weight_dst) / count
	}

def merge_boundary(graph, src, dst):
	"""Call back called before merging 2 nodes.

	In this case we don't need to do any computation here.
	"""
	pass

# REF [site] >> https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_boundary_merge.html
def hierarchical_merging_of_region_boundary_rags_example():
	#img = data.coffee()

	edges = sobel(skimage.color.rgb2gray(img))

	labels = slic(img, compactness=30, n_segments=400)
	g = graph.rag_boundary(labels, edges)

	graph.show_rag(labels, g, img)
	plt.title('Initial RAG')

	labels2 = graph.merge_hierarchical(
		labels, g, thresh=0.08, rag_copy=False,
		in_place_merge=True,
		merge_func=merge_boundary,
		weight_func=weight_boundary
	)

	graph.show_rag(labels, g, img)
	plt.title('RAG after hierarchical merging')

	plt.figure()
	out = skimage.color.label2rgb(labels2, img, kind='avg')
	plt.imshow(out)
	plt.title('Final segmentation')

	plt.show()

def main():
	watershed_example()

	#active_contour_example()
	#comparison_of_segmentation_and_superpixel_algorithms_example()

	#rag_merging_example()
	#hierarchical_merging_of_region_boundary_rags_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
