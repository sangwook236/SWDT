#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import networkx as nx

# REF [site] >> https://networkx.github.io/documentation/latest/tutorial.html
def basic_operation_tutorial():
	# Create a graph.
	G = nx.Graph()

	# Nodes.
	G.add_node(1)
	G.add_nodes_from([2, 3])

	H = nx.path_graph(10)  # Creates a graph.
	G.add_nodes_from(H)
	G.add_node(H)

	#print('G.nodes = {}.'.format(G.nodes))
	print('G.nodes = {}.'.format(list(G.nodes)))

	# Edges.
	G.add_edge(1, 2)
	e = (2, 3)
	G.add_edge(*e)  # Unpack edge tuple.

	G.add_edges_from([(1, 2), (1, 3)])

	G.add_edges_from(H.edges)

	#print('G.edges = {}.'.format(G.edges))
	print('G.edges = {}.'.format(list(G.edges)))

	# Remove all nodes and edges.
	G.clear()

	#--------------------
	G.add_edges_from([(1, 2), (1, 3)])
	G.add_node(1)
	G.add_edge(1, 2)
	G.add_node('spam')  # Adds node 'spam'.
	G.add_nodes_from('spam')  # Adds 4 nodes: 's', 'p', 'a', 'm'.
	G.add_edge(3, 'm')

	print('G.number_of_nodes() = {}.'.format(G.number_of_nodes()))
	print('G.number_of_edges() = {}.'.format(G.number_of_edges()))

	# Set-like views of the nodes, edges, neighbors (adjacencies), and degrees of nodes in a graph.
	print('G.adj[1] = {}.'.format(list(G.adj[1])))  # or G.neighbors(1).
	print('G.degree[1] = {}.'.format(G.degree[1]))  # The number of edges incident to 1.

	# Report the edges and degree from a subset of all nodes using an nbunch.
	# An nbunch is any of: None (meaning all nodes), a node, or an iterable container of nodes that is not itself a node in the graph.
	print("G.edges([2, 'm']) = {}.".format(G.edges([2, 'm'])))
	print('G.degree([2, 3]) = {}.'.format(G.degree([2, 3])))

	# Remove nodes and edges from the graph in a similar fashion to adding.
	G.remove_node(2)
	G.remove_nodes_from('spam')
	print('G.nodes = {}.'.format(list(G.nodes)))
	G.remove_edge(1, 3)

	# When creating a graph structure by instantiating one of the graph classes you can specify data in several formats.
	G.add_edge(1, 2)
	H = nx.DiGraph(G)  # Creates a DiGraph using the connections from G.
	print('H.edges() = {}.'.format(list(H.edges())))

	edgelist = [(0, 1), (1, 2), (2, 3)]
	H = nx.Graph(edgelist)

	#--------------------
	# Access edges and neighbors.
	print('G[1] = {}.'.format(G[1]))  # Same as G.adj[1].
	print('G[1][2] = {}.'.format(G[1][2]))  # Edge 1-2.
	print('G.edges[1, 2] = {}.'.format(G.edges[1, 2]))

	# Get/set the attributes of an edge using subscript notation if the edge already exists.
	G.add_edge(1, 3)
	G[1][3]['color'] = 'blue'
	G.edges[1, 2]['color'] = 'red'

	# Fast examination of all (node, adjacency) pairs is achieved using G.adjacency(), or G.adj.items().
	# Note that for undirected graphs, adjacency iteration sees each edge twice.
	FG = nx.Graph()
	FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
	for n, nbrs in FG.adj.items():
		for nbr, eattr in nbrs.items():
			wt = eattr['weight']
			if wt < 0.5: print(f'({n}, {nbr}, {wt:.3})')

	# Convenient access to all edges is achieved with the edges property.
	for (u, v, wt) in FG.edges.data('weight'):
		if wt < 0.5: print(f'({u}, {v}, {wt:.3})')

	#--------------------
	# Attributes.

	# Graph attributes.
	G = nx.Graph(day='Friday')
	print('G.graph = {}.'.format(G.graph))

	G.graph['day'] = 'Monday'

	# Node attributes: add_node(), add_nodes_from(), or G.nodes.
	G.add_node(1, time='5pm')
	G.add_nodes_from([3], time='2pm')
	print('G.nodes[1] = {}.'.format(G.nodes[1]))
	G.nodes[1]['room'] = 714
	print('G.nodes.data() = {}.'.format(G.nodes.data()))

	print('G.nodes[1] = {}.'.format(G.nodes[1]))  # List the attributes of a node.
	print('G.nodes[1].keys() = {}.'.format(G.nodes[1].keys()))
	#print('G[1] = {}.'.format(G[1]))  # G[1] = G.adj[1].

	# Edge attributes: add_edge(), add_edges_from(), or subscript notation.
	G.add_edge(1, 2, weight=4.7)
	G.add_edges_from([(3, 4), (4, 5)], color='red')
	G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
	G[1][2]['weight'] = 4.7
	G.edges[3, 4]['weight'] = 4.2
	print('G.edges.data() = {}.'.format(G.edges.data()))

	print('G.edges[3, 4] = {}.'.format(G.edges[3, 4]))  # List the attributes of an edge.
	print('G.edges[3, 4].keys() = {}.'.format(G.edges[3, 4].keys()))

	#--------------------
	# Directed graphs.

	DG = nx.DiGraph()
	DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
	print("DG.out_degree(1, weight='weight') = {}.".format(DG.out_degree(1, weight='weight')))
	print("DG.degree(1, weight='weight') = {}.".format(DG.degree(1, weight='weight')))  # The sum of in_degree() and out_degree().
	print('DG.successors(1) = {}.'.format(list(DG.successors(1))))
	print('DG.neighbors(1) = {}.'.format(list(DG.neighbors(1))))

	# Convert G to undirected graph.
	#H = DG.to_undirected()
	H = nx.Graph(DG)

	#--------------------
	# Multigraphs: Graphs which allow multiple edges between any pair of nodes.

	MG = nx.MultiGraph()
	#MDG = nx.MultiDiGraph()
	MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
	print("MG.degree(weight='weight') = {}.".format(dict(MG.degree(weight='weight'))))

	GG = nx.Graph()
	for n, nbrs in MG.adjacency():
		for nbr, edict in nbrs.items():
			minvalue = min([d['weight'] for d in edict.values()])
			GG.add_edge(n, nbr, weight = minvalue)
	print('nx.shortest_path(GG, 1, 3) = {}.'.format(nx.shortest_path(GG, 1, 3)))

	#--------------------
	# Classic graph operations:

	"""
	subgraph(G, nbunch):		induced subgraph view of G on nodes in nbunch
	union(G1,G2):				graph union
	disjoint_union(G1,G2):		graph union assuming all nodes are different
	cartesian_product(G1,G2):	return Cartesian product graph
	compose(G1,G2):				combine graphs identifying nodes common to both
	complement(G):				graph complement
	create_empty_copy(G):		return an empty copy of the same graph class
	to_undirected(G):			return an undirected representation of G
	to_directed(G):				return a directed representation of G
	"""

	#--------------------
	# Graph generators.

	# Use a call to one of the classic small graphs:
	petersen = nx.petersen_graph()
	tutte = nx.tutte_graph()
	maze = nx.sedgewick_maze_graph()
	tet = nx.tetrahedral_graph()

	# Use a (constructive) generator for a classic graph:
	K_5 = nx.complete_graph(5)
	K_3_5 = nx.complete_bipartite_graph(3, 5)
	barbell = nx.barbell_graph(10, 10)
	lollipop = nx.lollipop_graph(10, 20)

	# Use a stochastic graph generator:
	er = nx.erdos_renyi_graph(100, 0.15)
	ws = nx.watts_strogatz_graph(30, 3, 0.1)
	ba = nx.barabasi_albert_graph(100, 5)
	red = nx.random_lobster(100, 0.9, 0.9)

	#--------------------
	# Read a graph stored in a file using common graph formats, such as edge lists, adjacency lists, GML, GraphML, pickle, LEDA and others.

	nx.write_gml(red, './test.gml')
	mygraph = nx.read_gml('./test.gml')

# REF [site] >> https://networkx.github.io/documentation/latest/tutorial.html
def drawing_tutorial():
	import matplotlib.pyplot as plt

	G = nx.petersen_graph()
	plt.subplot(121)
	nx.draw(G, with_labels=True, font_weight='bold')
	plt.subplot(122)
	nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
	plt.show()

	options = {
		'node_color': 'black',
		'node_size': 100,
		'width': 3,
	}
	plt.subplot(221)
	nx.draw_random(G, **options)
	plt.subplot(222)
	#nx.draw_planar(G, **options)
	nx.draw_circular(G, **options)
	plt.subplot(223)
	nx.draw_spectral(G, **options)
	#nx.draw_spring(G, **options)
	#nx.draw_kamada_kawai(G, **options)
	plt.subplot(224)
	nx.draw_shell(G, nlist=[range(5, 10), range(5)], **options)
	plt.show()

	G = nx.dodecahedral_graph()
	shells = [[2, 3, 4, 5, 6], [8, 1, 0, 19, 18, 17, 16, 15, 14, 7], [9, 10, 11, 12, 13]]
	nx.draw_shell(G, nlist=shells, **options)
	plt.show()

	# Save drawings to a file.
	nx.draw(G)
	plt.savefig('./path.png')

	# If Graphviz and PyGraphviz or pydot are available on your system,
	# you can also use nx_agraph.graphviz_layout(G) or nx_pydot.graphviz_layout(G) to get the node positions,
	# or write the graph in dot format for further processing.
	pos = nx.nx_agraph.graphviz_layout(G)  # e.g.) pos = {1: (10, 10), 2: (30, 20)}.
	nx.draw(G, pos=pos)
	nx.drawing.nx_pydot.write_dot(G, './file.dot')

	#--------------------
	G = nx.complete_graph(15)

	#pos = nx.get_node_attributes(G, 'pos')
	#pos = nx.nx_agraph.graphviz_layout(G)
	#pos = nx.drawing.nx_pydot.graphviz_layout(G)
	#pos = nx.drawing.nx_pydot.pydot_layout(G)
	#pos = nx.random_layout(G)
	#pos = nx.planar_layout(G)
	#pos = nx.circular_layout(G)
	pos = nx.spectral_layout(G)
	#pos = nx.spiral_layout(G)
	#pos = nx.spring_layout(G)
	#pos = nx.shell_layout(G)
	#pos = nx.kamada_kawai_layout(G)
	#pos = nx.rescale_layout(G)
	#pos = nx.rescale_layout_dict(G)
	#pos = nx.bipartite_layout(G)
	#pos = nx.multipartite_layout(G)

	plt.figure(figsize=(10, 6))
	nx.draw_networkx_nodes(G, pos, node_size=400, alpha=1.0, node_shape='o', node_color='red')
	nx.draw_networkx_edges(G, pos, width=5, alpha=0.8, edge_color='blue')
	#nx.draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal')
	#nx.draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal')
	plt.tight_layout()
	plt.axis('off')
	#plt.savefig('./graph_drawing_1.svg')

	plt.figure(figsize=(10, 6))
	#nx.draw(G, pos, ax=None)
	#nx.draw(G, pos, labels=node_labels, **options)
	#nx.draw(G, pos, labels=nx.get_node_attributes(G, 'node_labels'), **options)
	nx.draw_networkx(G, pos, arrows=None, with_labels=True)
	plt.tight_layout()
	plt.axis('off')
	#plt.savefig('./graph_drawing_2.svg')

	plt.show()

def main():
	# GPU acceleration:
	#	https://developer.nvidia.com/blog/7-drop-in-replacements-to-instantly-speed-up-your-python-data-science-workflows

	#basic_operation_tutorial()
	drawing_tutorial()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
