#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time

# Counts cycles of length N in an undirected and connected graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def count_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, vert, start, count):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1:
				count = count + 1
				return count
			else:
				return count

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[vert][i] == 1:
				# DFS for searching path by decreasing length by 1.
				count = DFS(graph, marked, n - 1, i, start, count)

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False
		return count

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	count = 0
	for i in range(V - (n - 1)):
		count = DFS(graph, marked, n - 1, i, i, count)

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return count // 2

# Finds cycles of length N in an undirected and connected graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles(graph, n):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, marked, n, vert, start, cycles, curr_path):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1 and \
				[curr_path[0]] + curr_path[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
				cycles.append(curr_path.copy())
			return

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[vert][i] == 1:
				curr_path.append(i)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, marked, n - 1, i, start, cycles, curr_path)

				curr_path.pop()

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, marked, n - 1, i, i, cycles, curr_path=[i])

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles

# Finds cycles of length N in an undirected and connected graph.
#	REF [site] >> https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
def find_cycles_in_graph_with_collinear_vertices(graph, n, collinear_vertex_sets):
	# Number of vertices.
	V = len(graph)

	def DFS(graph, collinear_vertex_sets, marked, n, vert, start, cycles, curr_path, collinear_flags):
		# Mark the vertex vert as visited.
		marked[vert] = True

		# If the path of length (n - 1) is found.
		if n == 0:
			# Mark vert as un-visited to make it usable again.
			marked[vert] = False

			# Check if vertex vert can end with vertex start.
			if graph[vert][start] == 1:
				if len(collinear_flags) > n:  # Collinear vertices exist in the cycle.
					new_cycle = list(vtx for flag, vtx in zip(collinear_flags[1:], curr_path) if not flag) + [curr_path[-1]]
					if [new_cycle[0]] + new_cycle[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
						cycles.append(new_cycle)
				else:
					if [curr_path[0]] + curr_path[1:][::-1] not in cycles:  # Check if the reversed current path exists in the set of cycles or not.
						cycles.append(curr_path.copy())
			return

		# For searching every possible path of length (n - 1).
		for i in range(V):
			if marked[i] == False and graph[vert][i] == 1:
				curr_path.append(i)
				is_collinear = False
				if len(curr_path) == 3:
					vertex_set = set(curr_path)
					for collinear_vertex_set in collinear_vertex_sets:
						if vertex_set.issubset(collinear_vertex_set):
							is_collinear = True
							break
				elif len(curr_path) > 3:
					for vertex_set in [set(curr_path[-3:]), set(curr_path[-2:] + [curr_path[0]])]:
						for collinear_vertex_set in collinear_vertex_sets:
							if vertex_set.issubset(collinear_vertex_set):  # It is sensitive to the order of vertices in curr_path.
						#for collinear_vertex_set in collinear_vertex_sets:
						#	if len(collinear_vertex_set.intersection(vertex_set)) > 2:  # Error.
								is_collinear = True
								break
						if is_collinear: break
				collinear_flags.append(is_collinear)

				# DFS for searching path by decreasing length by 1.
				DFS(graph, collinear_vertex_sets, marked, n if is_collinear else n - 1, i, start, cycles, curr_path, collinear_flags)

				curr_path.pop()
				collinear_flags.pop()

		# Marking vert as unvisited to make it usable again.
		marked[vert] = False

	# All vertex are marked un-visited initially.
	marked = [False] * V

	# Searching for cycle by using v - n + 1 vertices.
	cycles = list()
	for i in range(V - (n - 1)):
		DFS(graph, collinear_vertex_sets, marked, n - 1, i, i, cycles, curr_path=[i], collinear_flags=[False])

		# i-th vertex is marked as visited and will not be visited again.
		marked[i] = True
	
	return cycles

def main():
	# NOTE [info] >> Find n-cycles in a graph ==> Polygon and quadrilateral detection.
	#	(1) Find polygons from a set of line segments.
	#		Find cycles in a graph.
	#			Build graph with line segment ends and intersection points as vertices and line segments as edges, then find cycles using DFS.
	#			https://stackoverflow.com/questions/41245408/how-to-find-polygons-in-a-given-set-of-points-and-edges
	#			"Finding and Counting Given Length Cycles", Algorithmica 1997.
	#			https://www.geeksforgeeks.org/cycles-of-length-n-in-an-undirected-and-connected-graph/
	#	(2) Finds polygons from a set of infinite lines (a simplified version of (1)).
	#		Find all possible combinations of adjacent atomic polygons constructed by the infinite lines
	#	(3) Find quadilaterals from a set of line segments (a simplified version of (1)).
	#		Find cycles with four edges (4-cycles) in a graph.
	#	(4) Find quadilaterals from a set of infinite lines (a simplified version of (2) & (3)).
	#		https://stackoverflow.com/questions/45248205/finding-all-quadrilaterals-in-a-set-of-intersections

	if False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = None
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0],
			[1, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 0, 0],
			[1, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5}
		]
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[0, 1, 0, 1, 0, 0, 1],
			[0, 0, 0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6}
		]
	elif False:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 1],
			[0, 1, 0, 1, 0, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6}
		]
	elif True:
		# Adjacency matrix.
		graph = [
			[0, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 1, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[1, 0, 1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 1, 1],
			[0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 0, 0, 0, 1, 1, 0]
		]
		collinear_vertex_sets = [
			{0, 1, 5},
			{3, 4, 6, 7}
		]

	n = 4  # n-cycle.

	print('Start counting n-cycles in a graph...')
	start_time = time.time()
	num_cycles = count_cycles(graph, n)
	print('End counting n-cycles in a graph: {} secs.'.format(time.time() - start_time))
	print("#cycles of length {} = {}.".format(n, num_cycles))

	print('Start finding n-cycles in a graph...')
	start_time = time.time()
	cycles = find_cycles(graph, n)
	print('End finding n-cycles in a graph: {} secs.'.format(time.time() - start_time))
	print("Cycles of length {} = {}.".format(n, cycles))

	if collinear_vertex_sets:
		print('Start finding n-cycles in a graph with collinear vertices...')
		start_time = time.time()
		cycles = find_cycles_in_graph_with_collinear_vertices(graph, n, collinear_vertex_sets)
		print('End finding n-cycles in a graph with collinear vertices: {} secs.'.format(time.time() - start_time))
		print("Cycles of length {} = {}.".format(n, cycles))

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
