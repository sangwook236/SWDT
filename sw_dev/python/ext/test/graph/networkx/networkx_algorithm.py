#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import networkx as nx

# REF [site] >> https://networkx.github.io/documentation/latest/tutorial.html
def algorithm_tutorial():
	G = nx.Graph()
	G.add_edges_from([(1, 2), (1, 3)])
	G.add_node('spam')  # Adds node 'spam'.
	print('nx.connected_components(G) =', list(nx.connected_components(G)))
	print('sorted(d for n, d in G.degree()) =', sorted(d for n, d in G.degree()))
	print('nx.clustering(G) =', nx.clustering(G))

	sp = dict(nx.all_pairs_shortest_path(G))
	print('sp[3] =', sp[3])

# REF [site] >> https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.minimum_cut.html
def minimum_cut_example():
	G = nx.Graph()
	G.add_edge('x', 'a', capacity=3.0)
	G.add_edge('x', 'b', capacity=1.0)
	G.add_edge('a', 'c', capacity=3.0)
	G.add_edge('b', 'c', capacity=5.0)
	G.add_edge('b', 'd', capacity=4.0)
	G.add_edge('d', 'e', capacity=2.0)
	G.add_edge('c', 'y', capacity=2.0)
	G.add_edge('e', 'y', capacity=3.0)

	cut_value, partition = nx.minimum_cut(G, 'x', 'y')
	#cut_value, partition = nx.minimum_cut(G, 'x', 'y', flow_func=nx.algorithms.flow.shortest_augmenting_path)
	cut1, cut2 = partition
	print('Cut value = {}.'.format(cut_value))
	print('Cut #1 = {}.'.format(cut1))
	print('Cut #2 = {}.'.format(cut2))

	cutset = set()
	for u, nbrs in ((n, G[n]) for n in cut1):
		cutset.update((u, v) for v in nbrs if v in cut2)
	print('Cutset = {}.'.format(sorted(cutset)))
	assert cut_value == sum(G.edges[u, v]['capacity'] for (u, v) in cutset)

	#--------------------
	G = nx.DiGraph()
	G.add_edge('x', 'a', capacity=3.0)
	G.add_edge('x', 'b', capacity=1.0)
	G.add_edge('a', 'c', capacity=3.0)
	G.add_edge('b', 'c', capacity=5.0)
	G.add_edge('b', 'd', capacity=4.0)
	G.add_edge('d', 'e', capacity=2.0)
	G.add_edge('c', 'y', capacity=2.0)
	G.add_edge('e', 'y', capacity=3.0)

	cut_value, partition = nx.minimum_cut(G, 'x', 'y')
	#cut_value, partition = nx.minimum_cut(G, 'x', 'y', flow_func=nx.algorithms.flow.shortest_augmenting_path)
	cut1, cut2 = partition

	print('Cut value = {}.'.format(cut_value))
	print('Cut #1 = {}.'.format(cut1))
	print('Cut #2 = {}.'.format(cut2))

	cutset = set()
	for u, nbrs in ((n, G[n]) for n in cut1):
		cutset.update((u, v) for v in nbrs if v in cut2)
	print('Cutset = {}.'.format(sorted(cutset)))  # TODO [check] >> I think it's not a complete cutset.
	assert cut_value == sum(G.edges[u, v]['capacity'] for (u, v) in cutset)

def maximum_flow_example():
	G = nx.DiGraph()
	G.add_edge('x', 'a', capacity=3.0)
	G.add_edge('x', 'b', capacity=1.0)
	G.add_edge('a', 'c', capacity=3.0)
	G.add_edge('b', 'c', capacity=5.0)
	G.add_edge('b', 'd', capacity=4.0)
	G.add_edge('d', 'e', capacity=2.0)
	G.add_edge('c', 'y', capacity=2.0)
	G.add_edge('e', 'y', capacity=3.0)

	#--------------------
	# REF [site] >> https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.maximum_flow.html

	flow_value, flow_dict = nx.maximum_flow(G, 'x', 'y')
	#flow_value, flow_dict = nx.maximum_flow(G, 'x', 'y', flow_func=nx.algorithms.flow.shortest_augmenting_path)

	print('Flow value = {}.'.format(flow_value))
	print('Flow (x -> b) = {}.'.format(flow_dict['x']['b']))

	#--------------------
	# REF [site] >>
	#	https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.edmonds_karp.html
	#	https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.shortest_augmenting_path.html

	R = nx.algorithms.flow.edmonds_karp(G, 'x', 'y')
	#R = nx.algorithms.flow.shortest_augmenting_path(G, 'x', 'y')
	print('Residual network: {}.'.format(R.graph))

	flow_value = R.graph['flow_value']
	assert flow_value == nx.maximum_flow_value(G, 'x', 'y')

	print('Flow value = {}.'.format(flow_value))
	print('Flow (x -> b) = {}.'.format(R['x']['b']['flow']))
	print('Capacity (x -> b) = {}.'.format(R['x']['b']['capacity']))

	#--------------------
	# REF [site] >> https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.boykov_kolmogorov.html

	R = nx.algorithms.flow.boykov_kolmogorov(G, 'x', 'y')
	print('Residual network: {}.'.format(R.graph))

	flow_value = R.graph['flow_value']
	assert flow_value == nx.maximum_flow_value(G, 'x', 'y')

	print('Flow value = {}.'.format(flow_value))
	print('Flow (x -> b) = {}.'.format(R['x']['b']['flow']))
	print('Capacity (x -> b) = {}.'.format(R['x']['b']['capacity']))

	# A nice feature of the Boykov-Kolmogorov algorithm is that a partition of the nodes that defines a minimum cut can be easily computed based on the search trees used during the algorithm.
	source_tree, target_tree = R.graph['trees']
	partition = (set(source_tree), set(G) - set(source_tree))
	#partition = (set(G) - set(target_tree), set(target_tree))

	print('Source tree: {}.'.format(source_tree))
	print('Target tree: {}.'.format(target_tree))

	#--------------------
	# REF [site] >> https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.preflow_push.html

	R = nx.algorithms.flow.preflow_push(G, 'x', 'y')
	#R = nx.algorithms.flow.preflow_push(G, 'x', 'y', value_only=True)
	print('Residual network: {}.'.format(R.graph))

	flow_value = R.graph['flow_value']
	assert flow_value == nx.maximum_flow_value(G, 'x', 'y')

	print('Flow value = {}.'.format(flow_value))
	print('Flow (x -> b) = {}.'.format(R['x']['b']['flow']))
	print('Capacity (x -> b) = {}.'.format(R['x']['b']['capacity']))

	# preflow_push also stores the maximum flow value in the excess attribute of the sink node t.
	assert flow_value == R.nodes['y']['excess']

def main():
	#algorithm_tutorial()

	#minimun_cut_example()
	maximum_flow_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
