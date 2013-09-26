#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/utility.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <utility>
#include <algorithm>
#include <iostream>
#include <string>
#include <stdexcept>


namespace {
namespace local {

template <class Graph>
struct exercise_vertex
{
	exercise_vertex(Graph &g, const char *name)
	: g_(g), name_(name)
	{}

	typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor_type;

	void operator()(const vertex_descriptor_type &v) const
	{
		typename boost::property_map<Graph, boost::vertex_index_t>::type vertex_id = boost::get(boost::vertex_index, g_);

		std::cout << "vertex: " << name_[boost::get(vertex_id, v)] << std::endl;

		// write out the outgoing edges
		std::cout << "\tout-edges: ";
		typename boost::graph_traits<Graph>::out_edge_iterator out_i, out_end;
		typename boost::graph_traits<Graph>::edge_descriptor e;
		for (boost::tie(out_i, out_end) = boost::out_edges(v, g_); out_i != out_end; ++out_i)
		{
			e = *out_i;
			vertex_descriptor_type src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_id, src)] << "," << name_[boost::get(vertex_id, targ)] << ") ";
		}
		std::cout << std::endl;

		// write out the incoming edges
		std::cout << "\tin-edges: ";
		typename boost::graph_traits<Graph>::in_edge_iterator in_i, in_end;
		for (boost::tie(in_i, in_end) = boost::in_edges(v, g_); in_i != in_end; ++in_i)
		{
			e = *in_i;
			vertex_descriptor_type src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_id, src)] << "," << name_[boost::get(vertex_id, targ)] << ") ";
		}
		std::cout << std::endl;

		// write out all adjacent vertices
		std::cout << "\tadjacent vertices: ";
		typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
		for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, g_); ai != ai_end; ++ai)
			std::cout << name_[boost::get(vertex_id, *ai)] <<  " ";
		std::cout << std::endl;
	}

private:
	Graph &g_;
	const char *name_;
};

// [ref] ${BOOST_HOME}/libs/graph/example/quick_tour.cpp
void boost_quick_tour()
{
	// create a typedef for the graph_type type
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_weight_t, float> > graph_type;

	// make convenient labels for the vertices
	enum { A, B, C, D, E, N };
	const int num_vertices = N;
	const char name[] = "ABCDE";

	// writing out the edges in the graph
	typedef std::pair<int, int> edge_type;

	const edge_type edge_array[] = { edge_type(A, B), edge_type(A, D), edge_type(C, A), edge_type(D, C), edge_type(C, E), edge_type(B, D), edge_type(D, E), };
	const int num_edges = sizeof(edge_array) / sizeof(edge_array[0]);

	// average transmission delay (in milliseconds) for each connection
	const float transmission_delay[] = { 1.2f, 4.5f, 2.6f, 0.4f, 5.2f, 1.8f, 3.3f, 9.1f };

	// declare a graph object, adding the edges and edge properties
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// VC++ can't handle the iterator constructor
	graph_type g(num_vertices);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
	for (std::size_t j = 0; j < num_edges; ++j)
	{
		boost::graph_traits<graph_type>::edge_descriptor e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = transmission_delay[j];
	}
#else
	graph_type g(edge_array, edge_array + num_edges, transmission_delay, num_vertices);
#endif

	boost::property_map<graph_type, boost::vertex_index_t>::type vertex_ids = boost::get(boost::vertex_index, g);
	boost::property_map<graph_type, boost::edge_weight_t>::type trans_delays = boost::get(boost::edge_weight, g);

	std::cout << "vertices(g) = ";
	typedef boost::graph_traits<graph_type>::vertex_iterator vertex_iterator_type;
	std::pair<vertex_iterator_type, vertex_iterator_type> vp;
	for (vp = boost::vertices(g); vp.first != vp.second; ++vp.first)
	{
		std::cout << name[boost::get(vertex_ids, *vp.first)] <<  " ";
		//std::cout << name[vertex_ids[*vp.first]] <<  " ";
	}
	std::cout << std::endl;

	std::cout << "edges(g) = ";
	boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
	{
		std::cout << "(" << name[boost::get(vertex_ids, boost::source(*ei, g))] << "," << name[boost::get(vertex_ids, boost::target(*ei, g))] << ") ";
		//std::cout << "(" << name[vertex_ids[boost::source(*ei, g)]] << "," << name[vertex_ids[boost::target(*ei, g)]] << ") ";
	}
	std::cout << std::endl;

	//
	std::for_each(boost::vertices(g).first, boost::vertices(g).second, exercise_vertex<graph_type>(g, name));

	//
	std::map<std::string, std::string> graph_attr, vertex_attr, edge_attr;
	graph_attr["size"] = "3,3";
	graph_attr["rankdir"] = "LR";
	graph_attr["ratio"] = "fill";
	vertex_attr["shape"] = "circle";

	boost::write_graphviz(
		std::cout,
		g,
		boost::make_label_writer(name),
		boost::make_label_writer(trans_delays),
		boost::make_graph_attributes_writer(graph_attr, vertex_attr, edge_attr)
	);
}

// [ref] http://www.ibm.com/developerworks/aix/library/au-aix-boost-graph/index.html
void basic_operation()
{
	std::cout << "--------------------------------------------------------------" << std::endl;
	{
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> graph_type;

		// create a simple undirected graph
		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(0, 3, g);
		boost::add_edge(1, 2, g);
		boost::add_edge(2, 3, g);

		graph_type::vertex_iterator vertexIt, vertexEnd;  // iterate over all the vertices of the graph
		graph_type::adjacency_iterator neighbourIt, neighbourEnd;  // iterate over the corresponding adjacent vertices

		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << *vertexIt << " is connected with ";
			boost::tie(neighbourIt, neighbourEnd) = boost::adjacent_vertices(*vertexIt, g);
			for (; neighbourIt != neighbourEnd; ++neighbourIt)
				std::cout << *neighbourIt << " ";
			std::cout << std::endl;
		}
	}

	std::cout << "\n--------------------------------------------------------------" << std::endl;
	{
		// Instead of using the adjacency list-based version to create an undirected graph, you can use the BGL-provided undirected_graph class.
		// However, this class internally uses an adjacency list, and using graphs based on adjacency lists always provides for greater flexibility.

		boost::undirected_graph<> g;

		boost::undirected_graph<>::vertex_descriptor u = g.add_vertex();
		boost::undirected_graph<>::vertex_descriptor v = g.add_vertex();
		boost::undirected_graph<>::vertex_descriptor w = g.add_vertex();
		boost::undirected_graph<>::vertex_descriptor x = g.add_vertex();
		boost::add_edge(u, v, g);
		boost::add_edge(u, w, g);
		boost::add_edge(u, x, g);
		std::cout << "degree of u: " << boost::degree(u, g) << std::endl;
	}

	std::cout << "\n--------------------------------------------------------------" << std::endl;
	{
		// when using the directedS tag in BGL, you are allowed to use only the out_edges helper function and associated iterators.
		// 5sing in_edges requires changing the graph type to bidirectionalS, although this is still more or less a directed graph.
		// typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS> graph_type;  // compile-time error
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS> graph_type;

		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(0, 3, g);
		boost::add_edge(1, 2, g);
		boost::add_edge(2, 3, g);

		graph_type::vertex_iterator vertexIt, vertexEnd;
		graph_type::in_edge_iterator inedgeIt, inedgeEnd;

		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "incoming edges for " << *vertexIt << ": ";
			boost::tie(inedgeIt, inedgeEnd) = boost::in_edges(*vertexIt, g);
			for (; inedgeIt != inedgeEnd; ++inedgeIt)
				std::cout << *inedgeIt << " ";
			std::cout << std::endl;
		}

		//
		graph_type::out_edge_iterator outedgeIt, outedgeEnd;

		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "out-edges for " << *vertexIt << ": ";
			boost::tie(outedgeIt, outedgeEnd) = boost::out_edges(*vertexIt, g);  // similar to incoming edges
			for (; outedgeIt != outedgeEnd; ++outedgeIt)
				std::cout << *outedgeIt << " ";
			std::cout << std::endl;
		}
	}

	std::cout << "\n--------------------------------------------------------------" << std::endl;
	{
		typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty> graph_type;

		graph_type g;
		boost::add_edge(0, 1, 8, g);
		boost::add_edge(0, 3, 18, g);
		boost::add_edge(1, 2, 20, g);
		boost::add_edge(2, 3, 2, g);
		boost::add_edge(3, 1, 1, g);
		boost::add_edge(1, 3, 7, g);

		std::cout << "number of edges: " << boost::num_edges(g) << std::endl;
		std::cout << "number of vertices: " << boost::num_vertices(g) << std::endl;

		graph_type::vertex_iterator vertexIt, vertexEnd;
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "in-degree for " << *vertexIt << ": " << boost::in_degree(*vertexIt, g) << std::endl;
			std::cout << "out-degree for " << *vertexIt << ": " << boost::out_degree(*vertexIt, g) << std::endl;
		}

		graph_type::edge_iterator edgeIt, edgeEnd;
		boost::tie(edgeIt, edgeEnd) = boost::edges(g);
		for (; edgeIt != edgeEnd; ++edgeIt)
			std::cout << "edge " << boost::source(*edgeIt, g) << "-->" << boost::target(*edgeIt, g) << std::endl;
	}
}

class custom_dfs_visitor : public boost::default_dfs_visitor
{
public:
	// this is invoked when a vertex is encountered for the first time.
	template<typename Vertex, typename Graph>
	void discover_vertex(const Vertex &u, const Graph &g) const
	{ std::cout << "at " << u << std::endl; }

	// this is invoked on every outgoing edge from the vertex after the vertex is discovered.
	template<typename Edge, typename Graph>
	void examine_edge(const Edge &e, const Graph &g) const
	{ std::cout << "examining edges " << e << std::endl; }

/*
	// this is invoked on the source vertex before traversal begins
	template<typename Vertex, typename Graph>
	void start_vertex(Vertex u, const Graph &g) const
	{
		// do something
	}

	// this is invoked when a vertex is invoked for the first time
	template<typename Vertex, typename Graph>
	void discover_vertex(Vertex u, const Graph &g) const
	{
		// do something
	}

	// if u is the root of a tree, finish_vertex is invoked after the same is invoked on all other elements of the tree.
	// if u is a leaf, then this method is invoked after all outgoing edges from u have been examined.
	template<typename Vertex, typename Graph>
	void finish_vertex(Vertex u, const Graph &g) const
	{
		// do something
	}

	// this is nvoked on every outgoing edge of u after it is discovered
	template<typename Edge, typename Graph>
	void examine_edge(Edge e, const Graph &g) const
	{
		// do something
	}

	// this is invoked on an edge after it becomes a member of the edges that form the search tree
	template<typename Edge, typename Graph>
	void tree_edge(Edge e, const Graph &g) const
	{
		// do something
	}

	// this is invoked on the back edges of a graph; used for an undirected graph, and because (u, v) and (v, u) are the same edges, both tree_edge and back_edge are invoked
	template<typename Edge, typename Graph>
	void back_edge(Edge e, const Graph &g) const
	{
		// do something
	}
*/
};

class custom_bfs_visitor : public boost::default_bfs_visitor
{
    template <class Vertex, class Graph>
    boost::graph::bfs_visitor_event_not_overridden initialize_vertex(Vertex u, Graph &g)
    {
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Vertex, class Graph>
	boost::graph::bfs_visitor_event_not_overridden discover_vertex(Vertex u, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Vertex, class Graph>
	boost::graph::bfs_visitor_event_not_overridden examine_vertex(Vertex u, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Edge, class Graph>
	boost::graph::bfs_visitor_event_not_overridden examine_edge(Edge e, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Edge, class Graph>
	boost::graph::bfs_visitor_event_not_overridden tree_edge(Edge e, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Edge, class Graph>
	boost::graph::bfs_visitor_event_not_overridden non_tree_edge(Edge e, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Edge, class Graph>
	boost::graph::bfs_visitor_event_not_overridden gray_target(Edge e, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Edge, class Graph>
	boost::graph::bfs_visitor_event_not_overridden black_target(Edge e, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	template <class Vertex, class Graph>
	boost::graph::bfs_visitor_event_not_overridden finish_vertex(Vertex u, Graph &g)
	{
		// do something
		return boost::graph::bfs_visitor_event_not_overridden();
	}
};

// [ref] http://www.ibm.com/developerworks/aix/library/au-aix-boost-graph/index.html
void traversal()
{
	std::cout << "depth-first search -------------------------------------------" << std::endl;
	{
		typedef boost::property<boost::edge_weight_t, int> EdgeWeightProperty;
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, boost::no_property, EdgeWeightProperty> graph_type;

		graph_type g;
		boost::add_edge(0, 1, 8, g);
		boost::add_edge(0, 3, 18, g);
		boost::add_edge(1, 2, 20, g);
		boost::add_edge(2, 3, 2, g);
		boost::add_edge(3, 1, 1, g);
		boost::add_edge(1, 3, 7, g);

		custom_dfs_visitor vis;
		boost::depth_first_search(g, boost::visitor(vis));
	}

	std::cout << "\nbreadth-first search -----------------------------------------" << std::endl;
	{
	}
}

void other_core_algorithms()
{
	std::cout << "ntopological sort ---------------------------------------------" << std::endl;
	{
		//boost::topological_sort();
		throw std::runtime_error("not yet implemented");
	}

	std::cout << "\ntransitive closure -------------------------------------------" << std::endl;
	{
		//boost::transitive_closure();
		throw std::runtime_error("not yet implemented");
	}
}

// [ref] ${BOOST_HOME}/libs/graph/example/dijkstra-example.cpp
void dijkstra_example()
{
	typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;
	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
	typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
	typedef std::pair<int, int> edge_type;

	enum nodes { A, B, C, D, E };

	const int num_nodes = 5;
	const char name[] = "ABCDE";

	const edge_type edge_array[] = { edge_type(A, C), edge_type(B, B), edge_type(B, D), edge_type(B, E), edge_type(C, B), edge_type(C, D), edge_type(D, E), edge_type(E, A), edge_type(E, B) };
	const int edge_weights[] = { 1, 2, 1, 2, 7, 3, 1, 1, 1 };
	const int num_arcs = sizeof(edge_array) / sizeof(edge_type);

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	graph_type g(num_nodes);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
	for (std::size_t j = 0; j < num_arcs; ++j)
	{
		edge_descriptor_type e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = edge_weights[j];
	}
#else
	graph_type g(edge_array, edge_array + num_arcs, edge_weights, num_nodes);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
#endif

	std::vector<vertex_descriptor_type> p(boost::num_vertices(g));
	std::vector<int> d(boost::num_vertices(g));
	vertex_descriptor_type s = boost::vertex(A, g);

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// VC++ has trouble with the named parameters mechanism
	boost::property_map<graph_type, boost::vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	boost::dijkstra_shortest_paths(
		g, s, &p[0], &d[0], weightmap, indexmap,
		std::less<int>(), boost::closed_plus<int>(),
		(std::numeric_limits<int>::max)(), 0,
		boost::default_dijkstra_visitor()
	);
#else
	boost::dijkstra_shortest_paths(g, s, boost::predecessor_map(&p[0]).distance_map(&d[0]));
#endif

	std::cout << "distances and parents: " << std::endl;
	boost::graph_traits<graph_type>::vertex_iterator vi, vend;
	for (boost::tie(vi, vend) = boost::vertices(g); vi != vend; ++vi)
	{
		std::cout << "distance(" << name[*vi] << ") = " << d[*vi] << ", ";
		std::cout << "parent(" << name[*vi] << ") = " << name[p[*vi]] << std::endl;
	}
	std::cout << std::endl;

	//
	std::ofstream dot_file("data/boost/dijkstra-eg.dot");
	if (dot_file.is_open())
	{
		dot_file << "digraph D {\n"
			<< "  rankdir=LR\n"
			<< "  size=\"4,3\"\n"
			<< "  ratio=\"fill\"\n"
			<< "  edge[style=\"bold\"]\n"
			<< "  node[shape=\"circle\"]\n";

		boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		{
			edge_descriptor_type e = *ei;
			vertex_descriptor_type u = boost::source(e, g), v = boost::target(e, g);

			dot_file << "  " << name[u] << " -> " << name[v] << "[label=\"" << boost::get(weightmap, e) << "\"";
			if (p[v] == u)
				dot_file << ", color=\"black\"";
			else
				dot_file << ", color=\"grey\"";
			dot_file << "]" << std::endl;
		}
		dot_file << "}" << std::endl;
	}
}

template<class PredecessorMap>
class record_predecessors : public boost::default_dijkstra_visitor
{
public:
	record_predecessors(PredecessorMap p)
	: predecessor_(p)
	{}

	template<class Edge, class Graph>
	void edge_relaxed(Edge e, Graph &g)
	{
		// set the parent of the target(e) to source(e)
		boost::put(predecessor_, boost::target(e, g), boost::source(e, g));
	}

protected:
	PredecessorMap predecessor_;
};

template<class PredecessorMap>
record_predecessors<PredecessorMap> make_predecessor_recorder(PredecessorMap p)
{
	return record_predecessors<PredecessorMap>(p);
}

class custom_dijkstra_visitor : public boost::default_dijkstra_visitor
{
    template <class Edge, class Graph>
    void edge_relaxed(Edge e, Graph &g)
	{
		// do something
    }

    template <class Edge, class Graph>
    void edge_not_relaxed(Edge e, Graph &g)
	{
		// do something
    }
};

class custom_bellman_visitor : public boost::default_bellman_visitor
{
	template <class Edge, class Graph>
	void examine_edge(Edge u, Graph &g)
	{
		// do something
	}

	template <class Edge, class Graph>
	void edge_relaxed(Edge u, Graph &g)
	{
		// do something
	}

	template <class Edge, class Graph>
	void edge_not_relaxed(Edge u, Graph &g)
	{
		// do something
	}

	template <class Edge, class Graph>
	void edge_minimized(Edge u, Graph &g)
	{
		// do something
	}

	template <class Edge, class Graph>
	void edge_not_minimized(Edge u, Graph &g)
	{
		// do something
	}
};

class custom_astar_visitor : public boost::default_astar_visitor
{
	template <class Edge, class Graph>
	void edge_relaxed(Edge e, const Graph &g)
	{
		// do something
	}

	template <class Edge, class Graph>
	void edge_not_relaxed(Edge e, const Graph &g)
	{
		// do something
	}
};

// [ref] http://library.developer.nokia.com/index.jsp?topic=/S60_5th_Edition_Cpp_Developers_Library/GUID-02F20077-73B5-4A63-85DB-D909E0ADE01C/html/con_graph_quick_tour.html
void shortest_paths()
{
	// Dijkstra's algorithm -----------------------------------------
	dijkstra_example();

	{
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;
		typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
		typedef std::pair<int, int> edge_type;

		const int num_nodes = 5;
		const edge_type edges[] = { edge_type(0, 2), edge_type(1, 1), edge_type(1, 3), edge_type(1, 4), edge_type(2, 1), edge_type(2, 3), edge_type(3, 4), edge_type(4, 0), edge_type(4, 1) };
		const int edge_weights[] = { 1, 2, 1, 2, 7, 3, 1, 1, 1 };

		graph_type g(edges, edges + sizeof(edges) / sizeof(edge_type), edge_weights, num_nodes);

		// vector for storing distance property
		std::vector<int> d(boost::num_vertices(g));
		// get the first vertex
		vertex_descriptor_type s = *(boost::vertices(g).first);

		// invoke variant 2 of Dijkstra's algorithm
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]));

		std::cout << "distances from start vertex:" << std::endl;
		boost::graph_traits<graph_type>::vertex_iterator vi;
		for (vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
			std::cout << "distance(" << *vi << ") = " << d[*vi] << std::endl;
		std::cout << std::endl;

		//
		std::vector<vertex_descriptor_type> p(boost::num_vertices(g));  // the predecessor array
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]).visitor(make_predecessor_recorder(&p[0])));

		std::cout << "parents in the tree of shortest paths:" << std::endl;
		for (vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
		{
			std::cout << "parent(" << *vi;
			if (p[*vi] == vertex_descriptor_type() && *vi == s)
			//if (p[*vi] == boost::graph_traits<graph_type>::null_vertex())  // not working
				std::cout << ") = no parent" << std::endl;
			else
				std::cout << ") = " << p[*vi] << std::endl;
		}
	}
}

// [ref] ${BOOST_HOME}/libs/graph/example/kruskal-example.cpp
void kruskal_minimum_spanning_tree_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;
	typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
	typedef std::pair<int, int> edge_type;

	const int num_nodes = 5;
	const edge_type edge_array[] = { edge_type(0, 2), edge_type(1, 3), edge_type(1, 4), edge_type(2, 1), edge_type(2, 3), edge_type(3, 4), edge_type(4, 0), edge_type(4, 1) };
	const int weights[] = { 1, 1, 2, 7, 3, 1, 1, 1 };
	const std::size_t num_edges = sizeof(edge_array) / sizeof(edge_type);

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	graph_type g(num_nodes);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
	for (std::size_t j = 0; j < num_edges; ++j)
	{
		edge_descriptor_type e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = weights[j];
	}
#else
	graph_type g(edge_array, edge_array + num_edges, weights, num_nodes);
#endif
	boost::property_map<graph_type, boost::edge_weight_t>::type weight = boost::get(boost::edge_weight, g);
	std::vector<edge_descriptor_type> spanning_tree;

	boost::kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

	//
	std::cout << "print the edges in the MST:" << std::endl;
	for (std::vector < edge_descriptor_type >::iterator ei = spanning_tree.begin(); ei != spanning_tree.end(); ++ei)
	{
		std::cout << boost::source(*ei, g) << " <--> " << boost::target(*ei, g) << " with weight of " << weight[*ei] << std::endl;
	}

	//
	std::ofstream fout("./data/boost/kruskal-eg.dot");
	fout << "graph A {\n"
		<< " rankdir=LR\n"
		<< " size=\"3,3\"\n"
		<< " ratio=\"filled\"\n"
		<< " edge[style=\"bold\"]\n"
		<< " node[shape=\"circle\"]\n";
	boost::graph_traits<graph_type>::edge_iterator eiter, eiter_end;
	for (boost::tie(eiter, eiter_end) = boost::edges(g); eiter != eiter_end; ++eiter)
	{
		fout << boost::source(*eiter, g) << " -- " << boost::target(*eiter, g);
		if (std::find(spanning_tree.begin(), spanning_tree.end(), *eiter) != spanning_tree.end())
			fout << " [color=\"black\", label=\"" << boost::get(boost::edge_weight, g, *eiter) << "\"];" << std::endl;
		else
			fout << " [color=\"gray\", label=\"" << boost::get(boost::edge_weight, g, *eiter) << "\"];" << std::endl;
	}
	fout << '}' << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/kruskal-telephone.cpp
void kruskal_minimum_spanning_tree_telephone_example()
{
/*
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,boost::property<boost::edge_weight_t, int> > graph_type;

#if 0
	// [ref] GraphvizGraph was defined in boost/graph/graphviz.hpp, but cannot be used at present

	typedef std::map<std::string, std::string> GraphvizAttrList;
	typedef boost::property<boost::vertex_attribute_t, GraphvizAttrList> GraphvizVertexProperty;
	typedef boost::property<boost::edge_attribute_t, GraphvizAttrList, boost::property<boost::edge_index_t, int> > GraphvizEdgeProperty;
	typedef boost::property<
		boost::graph_graph_attribute_t, GraphvizAttrList,
		boost::property<boost::graph_vertex_attribute_t, GraphvizAttrList,
			boost::property<boost::graph_edge_attribute_t, GraphvizAttrList,
				boost::property<boost::graph_name_t, std::string> > >
	> GraphvizGraphProperty;
	typedef boost::subgraph<
		boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GraphvizVertexProperty, GraphvizEdgeProperty, GraphvizGraphProperty>
	> GraphvizGraph;
	typedef GraphvizGraph graphviz_graph_type;
#else
	typedef graph_type graphviz_graph_type;
#endif

	graphviz_graph_type g_dot;
	{
		// compile-time error

		std::ifstream stream("./data/boost/telephone-network.dot");
		const bool status = boost::read_graphviz(stream, g_dot);
	}

	graph_type g(boost::num_vertices(g_dot));
	boost::property_map<graphviz_graph_type, boost::edge_attribute_t>::type edge_attr_map = boost::get(boost::edge_attribute, g_dot);
	boost::graph_traits<graphviz_graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g_dot); ei != ei_end; ++ei)
	{
		const int weight = boost::lexical_cast<int>(edge_attr_map[*ei]["label"]);
		boost::property<boost::edge_weight_t, int> edge_property(weight);
		boost::add_edge(boost::source(*ei, g_dot), boost::target(*ei, g_dot), edge_property, g);
	}

	std::vector<boost::graph_traits<graph_type>::edge_descriptor> mst;
	typedef std::vector<boost::graph_traits<graph_type>::edge_descriptor>::size_type size_type;

	boost::kruskal_minimum_spanning_tree(g, std::back_inserter(mst));

	//
	boost::property_map<graph_type, boost::edge_weight_t>::type weight = boost::get(boost::edge_weight, g);
	int total_weight = 0;
	for (size_type e = 0; e < mst.size(); ++e)
		total_weight += boost::get(weight, mst[e]);
	std::cout << "total weight: " << total_weight << std::endl;

	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
	for (size_type i = 0; i < mst.size(); ++i)
	{
		const vertex_descriptor_type u = boost::source(mst[i], g), v = boost::target(mst[i], g);
		edge_attr_map[boost::edge(u, v, g_dot).first]["color"] = "black";
	}

	//
	std::ofstream out("./data/boost/telephone-mst-kruskal.dot");

	boost::graph_property<graphviz_graph_type, boost::graph_edge_attribute_t>::type &graph_edge_attr_map = boost::get_property(g_dot, graph_edge_attribute);
	graph_edge_attr_map["color"] = "gray";
	graph_edge_attr_map["style"] = "bold";

	boost::write_graphviz(out, g_dot);
*/
}

// [ref] ${BOOST_HOME}/libs/graph/example/prim-example.cpp
void prim_minimum_spanning_tree_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_distance_t, int>, boost::property<boost::edge_weight_t, int> > graph_type;
	typedef std::pair<int, int> edge_type;

	const int num_nodes = 5;
	const edge_type edges[] = { edge_type(0, 2), edge_type(1, 3), edge_type(1, 4), edge_type(2, 1), edge_type(2, 3), edge_type(3, 4), edge_type(4, 0) };
	const int weights[] = { 1, 1, 2, 7, 3, 1, 1 };

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	graph_type g(num_nodes);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
	for (std::size_t j = 0; j < sizeof(edges) / sizeof(edge_type); ++j)
	{
		boost::graph_traits<graph_type>::edge_descriptor e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edges[j].first, edges[j].second, g);
		weightmap[e] = weights[j];
	}
#else
	graph_type g(edges, edges + sizeof(edges) / sizeof(edge_type), weights, num_nodes);
	boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
#endif

	std::vector<boost::graph_traits<graph_type>::vertex_descriptor> p(boost::num_vertices(g));

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	boost::property_map<graph_type, boost::vertex_distance_t>::type distance = boost::get(boost::vertex_distance, g);
	boost::property_map<graph_type, boost::vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	boost::prim_minimum_spanning_tree(g, *boost::vertices(g).first, &p[0], distance, weightmap, indexmap, boost::default_dijkstra_visitor());
#else
	boost::prim_minimum_spanning_tree(g, &p[0]);
#endif

	//
	for (std::size_t i = 0; i != p.size(); ++i)
		if (p[i] != i)
			std::cout << "parent[" << i << "] = " << p[i] << std::endl;
		else
			std::cout << "parent[" << i << "] = no parent" << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/prim-telephone.cpp
void prim_minimum_spanning_tree_telephone_example()
{
/*
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;

#if 0
	// [ref] GraphvizGraph was defined in boost/graph/graphviz.hpp, but cannot be used at present

	typedef std::map<std::string, std::string> GraphvizAttrList;
	typedef boost::property<boost::vertex_attribute_t, GraphvizAttrList> GraphvizVertexProperty;
	typedef boost::property<boost::edge_attribute_t, GraphvizAttrList, boost::property<boost::edge_index_t, int> > GraphvizEdgeProperty;
	typedef boost::property<
		boost::graph_graph_attribute_t, GraphvizAttrList,
		boost::property<boost::graph_vertex_attribute_t, GraphvizAttrList,
			boost::property<boost::graph_edge_attribute_t, GraphvizAttrList,
				boost::property<boost::graph_name_t, std::string> > >
	> GraphvizGraphProperty;
	typedef boost::subgraph<
		boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, GraphvizVertexProperty, GraphvizEdgeProperty, GraphvizGraphProperty>
	> GraphvizGraph;
	typedef GraphvizGraph graphviz_graph_type;
#else
	typedef graph_type graphviz_graph_type;
#endif

	graphviz_graph_type g_dot;
	{
		// compile-time error

		std::ifstream stream("./data/boost/telephone-network.dot");
		const bool status = boost::read_graphviz(stream, g_dot);
	}

	graph_type g(boost::num_vertices(g_dot));
	boost::property_map<graphviz_graph_type, boost::edge_attribute_t>::type edge_attr_map = boost::get(boost::edge_attribute, g_dot);
	boost::graph_traits<graphviz_graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g_dot); ei != ei_end; ++ei)
	{
		const int weight = boost::lexical_cast<int>(edge_attr_map[*ei]["label"]);
		boost::property<boost::edge_weight_t, int> edge_property(weight);
		boost::add_edge(boost::source(*ei, g_dot), boost::target(*ei, g_dot), edge_property, g);
	}

	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
	std::vector<vertex_descriptor_type> parent(boost::num_vertices(g));
	boost::property_map<graph_type, boost::edge_weight_t>::type weight = boost::get(boost::edge_weight, g);
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	boost::property_map<graph_type, boost::vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	std::vector<std::size_t> distance(boost::num_vertices(g));
	boost::prim_minimum_spanning_tree(g, *boost::vertices(g).first, &parent[0], &distance[0], weight, indexmap, boost::default_dijkstra_visitor());
#else
	boost::prim_minimum_spanning_tree(g, &parent[0]);
#endif

	//
	int total_weight = 0;
	for (int v = 0; v < boost::num_vertices(g); ++v)
		if (parent[v] != v)
			total_weight += boost::get(weight, boost::edge(parent[v], v, g).first);
	std::cout << "total weight: " << total_weight << std::endl;

	for (int u = 0; u < boost::num_vertices(g); ++u)
		if (parent[u] != u)
			edge_attr_map[boost::edge(parent[u], u, g_dot).first]["color"] = "black";

	//
	std::ofstream out("./data/boost/telephone-mst-prim.dot");
	boost::graph_property<graphviz_graph_type, boost::graph_edge_attribute_t>::type &graph_edge_attr_map = boost::get_property(g_dot, boost::graph_edge_attribute);
	graph_edge_attr_map["color"] = "gray";
	boost::write_graphviz(out, g_dot);
*/
}

void minimum_spanning_tree()
{
	std::cout << "Kruskal's minimum spanning tree algorithm ---------------------" << std::endl;
	kruskal_minimum_spanning_tree_example();
	//kruskal_minimum_spanning_tree_telephone_example();  // compile-time error

	std::cout << "\nPrim's minimum spanning tree algorithm ------------------------" << std::endl;
	prim_minimum_spanning_tree_example();
	//prim_minimum_spanning_tree_telephone_example();  // compile-time error
}

void random_spanning_tree()
{
	throw std::runtime_error("not yet implemented");
}

void common_spanning_tree()
{
	throw std::runtime_error("not yet implemented");
}

// [ref] ${BOOST_HOME}/libs/graph/example/connected_components.cpp
// [ref] ${BOOST_HOME}/libs/graph/example/connected-components.cpp
void connected_components_algorithm()
{
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(1, 4, g);
		boost::add_edge(4, 0, g);
		boost::add_edge(2, 5, g);

		std::vector<int> components(boost::num_vertices(g));
		const int num = boost::connected_components(g, &components[0]);

		//
		std::cout << "total number of components: " << num << std::endl;
		for (std::vector<int>::size_type i = 0; i != components.size(); ++i)
		  std::cout << "vertex " << i <<" is in component " << components[i] << std::endl;
	}

	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

		const int N = 6;
		graph_type g(N);
		boost::add_edge(0, 1, g);
		boost::add_edge(1, 4, g);
		boost::add_edge(4, 0, g);
		boost::add_edge(2, 5, g);

		std::vector<int> c(boost::num_vertices(g));
		const int num = boost::connected_components(g, boost::make_iterator_property_map(c.begin(), boost::get(boost::vertex_index, g), c[0]));

		std::cout << std::endl;
		std::cout << "total number of components: " << num << std::endl;
		for (std::vector<int>::iterator i = c.begin(); i != c.end(); ++i)
			std::cout << "vertex " << i - c.begin() << " is in component " << *i << std::endl;
	}
}

// [ref] ${BOOST_HOME}/libs/graph/example/strong-components.cpp
// [ref] ${BOOST_HOME}/libs/graph/example/strong_components.cpp
void strong_components_algorithm()
{
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> graph_type;

		const int N = 6;
		graph_type g(N);
		boost::add_edge(0, 1, g);
		boost::add_edge(1, 1, g);
		boost::add_edge(1, 3, g);
		boost::add_edge(1, 4, g);
		boost::add_edge(3, 4, g);
		boost::add_edge(3, 0, g);
		boost::add_edge(4, 3, g);
		boost::add_edge(5, 2, g);

		std::vector<int> components(N);
		const int num = boost::strong_components(g, boost::make_iterator_property_map(components.begin(), boost::get(boost::vertex_index, g), components[0]));

		std::cout << "total number of components: " << num << std::endl;
		for (std::vector<int>::iterator i = components.begin(); i != components.end(); ++i)
			std::cout << "vertex " << i - components.begin() << " is in component " << *i << std::endl;
	}
/*
	{
		// vertex properties
		typedef boost::property<boost::vertex_name_t, std::string> vertex_p;
		// edge properties
		typedef boost::property<boost::edge_weight_t, double> edge_p;
		// graph properties
		typedef boost::property<boost::graph_name_t, std::string, boost::property<boost::graph_graph_attribute_t, float> > graph_p;
		// adjacency_list-based type
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_p, edge_p, graph_p> graph_type;
		//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> graph_type;

		graph_type g(0);
		boost::dynamic_properties dp;

		{
			// compile-time error

			boost::property_map<graph_type, boost::vertex_name_t>::type vertex_name = boost::get(boost::vertex_name, g);
			dp.property("shape", vertex_name);

			boost::property_map<graph_type, boost::graph_graph_attribute_t>::type graph_attr = boost::get(boost::graph_graph_attribute, g);
			//boost::ref_property_map<graph_type *, std::string> graph_attr(boost::get_property(g, boost::graph_graph_attribute));
			dp.property("ratio", graph_attr);

			std::ifstream stream("./data/boost/scc.dot");
			const bool status = boost::read_graphviz(stream, g, dp, "shape");
		}

		//
		const char *name = "abcdefghij";

		std::cout << "A directed graph:" << std::endl;
		boost::print_graph(g, name);
		std::cout << std::endl;

		typedef boost::graph_traits<boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> >::vertex_descriptor vertex_descriptor_type;

		std::vector<int> components(boost::num_vertices(g)), discover_time(boost::num_vertices(g));
		std::vector<boost::default_color_type> color(boost::num_vertices(g));
		std::vector<vertex_descriptor_type> root(boost::num_vertices(g));
		const int num = boost::strong_components(
			g,
			boost::make_iterator_property_map(components.begin(), boost::get(boost::vertex_index, g)),
			boost::root_map(boost::make_iterator_property_map(root.begin(), boost::get(boost::vertex_index, g))).
				color_map(boost::make_iterator_property_map(color.begin(), boost::get(boost::vertex_index, g))).
					discover_time_map(boost::make_iterator_property_map(discover_time.begin(), boost::get(boost::vertex_index, g)))
		);

		std::cout << "total number of components: " << num << std::endl;
		for (std::vector<int>::size_type i = 0; i != components.size(); ++i)
			std::cout << "vertex " << name[i] <<" is in component " << components[i] << std::endl;
	}
*/
}

struct edge_component_t
{
	enum { num = 555 };
	typedef boost::edge_property_tag kind;
} edge_component;

// [ref] ${BOOST_HOME}/libs/graph/example/biconnected_components.cpp
void biconnected_components_algorithm()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<edge_component_t, std::size_t> > graph_type;
	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;

	graph_type g(9);
	boost::add_edge(0, 5, g);
	boost::add_edge(0, 1, g);
	boost::add_edge(0, 6, g);
	boost::add_edge(1, 2, g);
	boost::add_edge(1, 3, g);
	boost::add_edge(1, 4, g);
	boost::add_edge(2, 3, g);
	boost::add_edge(4, 5, g);
	boost::add_edge(6, 8, g);
	boost::add_edge(6, 7, g);
	boost::add_edge(7, 8, g);

	//
	boost::property_map<graph_type, edge_component_t>::type components = boost::get(edge_component, g);
	const std::size_t num_comps = boost::biconnected_components(g, components);
	std::cout << "found " << num_comps << " biconnected components." << std::endl;

	//
	std::vector<vertex_descriptor_type> art_points;
	boost::articulation_points(g, std::back_inserter(art_points));
	std::cout << "found " << art_points.size() << " articulation points." << std::endl;

	//
	std::cout << "graph A {\n" << "  node[shape=\"circle\"]" << std::endl;
	for (std::size_t i = 0; i < art_points.size(); ++i)
		std::cout << (char)(art_points[i] + 'A') << " [ style=\"filled\", fillcolor=\"red\" ];" << std::endl;

	boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		std::cout << (char)(boost::source(*ei, g) + 'A') << " -- " << (char)(boost::target(*ei, g) + 'A') << "[label=\"" << components[*ei] << "\"]" << std::endl;
	std::cout << '}' << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/incremental-components-eg.cpp
// [ref] ${BOOST_HOME}/libs/graph/example/incremental_components.cpp
void incremental_connected_components_algorithm()
{
/*
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
		typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
		typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
		typedef boost::graph_traits<graph_type>::vertices_size_type vertex_index_type;

		// create a graph
		const int VERTEX_COUNT = 6;
		graph_type graph(VERTEX_COUNT);

		boost::add_edge(0, 1, graph);
		boost::add_edge(1, 4, graph);

		// create the disjoint-sets object, which requires rank and parent vertex properties.
		std::vector<vertex_descriptor_type> rank(boost::num_vertices(graph));
		std::vector<vertex_descriptor_type> parent(boost::num_vertices(graph));

		typedef vertex_index_type * rank_type;
		typedef vertex_descriptor_type * parent_type;
		boost::disjoint_sets<rank_type, parent_type> ds(&rank[0], &parent[0]);

		// determine the connected components, storing the results in the disjoint-sets object.
		boost::initialize_incremental_components(graph, ds);
		boost::incremental_components(graph, ds);

		// add a couple more edges and update the disjoint-sets
		boost::add_edge(4, 0, graph);
		boost::add_edge(2, 5, graph);

		ds.union_set(4, 0);
		ds.union_set(2, 5);

		BOOST_FOREACH(vertex_descriptor_type current_vertex, boost::vertices(graph))
			std::cout << "representative[" << current_vertex << "] = " << ds.find_set(current_vertex) << std::endl;
		std::cout << std::endl;

		// generate component index.
		// NOTE: We would need to pass in a vertex index map into the component_index constructor
		// if our graph type used listS instead of vecS (identity_property_map is used by default).
		typedef boost::component_index<vertex_index_type> components_type;
		components_type components(parent.begin(), parent.end());

		// iterate through the component indices
		BOOST_FOREACH(vertex_index_type component_index, components)
		{
			std::cout << "component " << component_index << " contains: ";

			// iterate through the child vertex indices for [component_index]
			BOOST_FOREACH(vertex_index_type child_index, components[component_index])
				std::cout << child_index << " ";
			std::cout << std::endl;
		}
	}
*/
/*
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
		typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
		typedef boost::graph_traits<graph_type>::vertices_size_type vertex_index_type;

		const int VERTEX_COUNT = 6;
		graph_type graph(VERTEX_COUNT);

		std::vector<vertex_index_type> rank(boost::num_vertices(graph));
		std::vector<vertex_descriptor_type> parent(boost::num_vertices(graph));

		typedef vertex_index_type * rank_type;
		typedef vertex_descriptor_type * parent_type;

		boost::disjoint_sets<rank_type, parent_type> ds(&rank[0], &parent[0]);

		boost::initialize_incremental_components(graph, ds);
		boost::incremental_components(graph, ds);

		boost::graph_traits<graph_type>::edge_descriptor edge;
		bool flag;

		boost::tie(edge, flag) = boost::add_edge(0, 1, graph);
		ds.union_set(0, 1);

		boost::tie(edge, flag) = boost::add_edge(1, 4, graph);
		ds.union_set(1, 4);

		boost::tie(edge, flag) = boost::add_edge(4, 0, graph);
		ds.union_set(4, 0);

		boost::tie(edge, flag) = boost::add_edge(2, 5, graph);
		ds.union_set(2, 5);

		std::cout << "an undirected graph:" << std::endl;
		boost::print_graph(graph, boost::get(boost::vertex_index, graph));
		std::cout << std::endl;

		BOOST_FOREACH(vertex_descriptor_type current_vertex, boost::vertices(graph))
			std::cout << "representative[" << current_vertex << "] = " << ds.find_set(current_vertex) << std::endl;
		std::cout << std::endl;

		typedef boost::component_index<vertex_index_type> components_type;

		// NOTE: Because we're using vecS for the graph type, we're effectively using identity_property_map for a vertex index map.
		// If we were to use listS instead, the index map would need to be explicitly passed to the component_index constructor.
		components_type components(parent.begin(), parent.end());

		// iterate through the component indices
		BOOST_FOREACH(vertex_index_type current_index, components)
		{
			std::cout << "component " << current_index << " contains: ";

			// iterate through the child vertex indices for [current_index]
			BOOST_FOREACH(vertex_index_type child_index, components[current_index])
				std::cout << child_index << " ";
			std::cout << std::endl;
		}
	}
*/
}

void connected_components()
{
	std::cout << "connected components algorithm --------------------------------" << std::endl;
	connected_components_algorithm();

	std::cout << "\nstrong components algorithm -----------------------------------" << std::endl;
	strong_components_algorithm();

	std::cout << "\nbiconnected components algorithm ------------------------------" << std::endl;
	biconnected_components_algorithm();

	std::cout << "\nincremental connected components algorithm --------------------" << std::endl;
	incremental_connected_components_algorithm();
}

// [ref] ${BOOST_HOME}/libs/graph/example/max_flow.cpp
void max_flow_example(std::istream &stream)
{
	typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits_type;
	typedef boost::adjacency_list<
		boost::listS, boost::vecS, boost::directedS,
		boost::property<boost::vertex_name_t, std::string>,
		boost::property<boost::edge_capacity_t, long,
		boost::property<boost::edge_residual_capacity_t, long,
		boost::property<boost::edge_reverse_t, traits_type::edge_descriptor> > >
	> graph_type;

	graph_type g;
	boost::property_map<graph_type, boost::edge_capacity_t>::type capacity = boost::get(boost::edge_capacity, g);
	boost::property_map<graph_type, boost::edge_reverse_t>::type reverse_edge = boost::get(boost::edge_reverse, g);
	boost::property_map<graph_type, boost::edge_residual_capacity_t>::type residual_capacity = boost::get(boost::edge_residual_capacity, g);

	traits_type::vertex_descriptor src, sink;
	boost::read_dimacs_max_flow(g, capacity, reverse_edge, src, sink, stream);

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// use non-named parameter version
	boost::property_map<graph_type, vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	const long flow = boost::push_relabel_max_flow(g, src, sink, capacity, residual_capacity, reverse_edge, indexmap);
#else
	const long flow = boost::push_relabel_max_flow(g, src, sink);
#endif

	std::cout << "c  The total flow:" << std::endl;
	std::cout << "s " << flow << std::endl << std::endl;

	std::cout << "c flow values:" << std::endl;
	boost::graph_traits<graph_type>::vertex_iterator u_iter, u_end;
	boost::graph_traits<graph_type>::out_edge_iterator ei, e_end;
	for (boost::tie(u_iter, u_end) = boost::vertices(g); u_iter != u_end; ++u_iter)
		for (boost::tie(ei, e_end) = boost::out_edges(*u_iter, g); ei != e_end; ++ei)
			if (capacity[*ei] > 0)
				std::cout << "f " << *u_iter << " " << boost::target(*ei, g) << " " << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/push-relabel-eg.cpp
void push_relabel_example(std::istream &stream)
{
	typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits_type;
	typedef boost::adjacency_list<
		boost::vecS, boost::vecS, boost::directedS,
		boost::property<boost::vertex_name_t, std::string>,
			boost::property<boost::edge_capacity_t, long,
				boost::property<boost::edge_residual_capacity_t, long,
					boost::property<boost::edge_reverse_t, traits_type::edge_descriptor> > >
	> graph_type;

	graph_type g;
	boost::property_map<graph_type, boost::edge_capacity_t>::type capacity = boost::get(boost::edge_capacity, g);
	boost::property_map<graph_type, boost::edge_residual_capacity_t >::type residual_capacity = boost::get(boost::edge_residual_capacity, g);
	boost::property_map<graph_type, boost::edge_reverse_t>::type reverse_edge = boost::get(boost::edge_reverse, g);
	traits_type::vertex_descriptor src, sink;
	boost::read_dimacs_max_flow(g, capacity, reverse_edge, src, sink, stream);

#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	boost::property_map<graph_type, boost::vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	const long flow = boost::push_relabel_max_flow(g, src, sink, capacity, residual_capacity, reverse_edge, indexmap);
#else
	const long flow = boost::push_relabel_max_flow(g, src, sink);
#endif

	std::cout << "c  The total flow:" << std::endl;
	std::cout << "s " << flow << std::endl << std::endl;

	std::cout << "c flow values:" << std::endl;
	boost::graph_traits<graph_type>::vertex_iterator u_iter, u_end;
	boost::graph_traits<graph_type>::out_edge_iterator ei, e_end;
	for (boost::tie(u_iter, u_end) = boost::vertices(g); u_iter != u_end; ++u_iter)
		for (boost::tie(ei, e_end) = boost::out_edges(*u_iter, g); ei != e_end; ++ei)
			if (capacity[*ei] > 0)
				std::cout << "f " << *u_iter << " " << boost::target(*ei, g) << " " << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/boykov_kolmogorov-eg.cpp
void boykov_kolmogorov_example(std::istream &stream)
{
	typedef boost::adjacency_list_traits<boost::vecS, boost::vecS, boost::directedS> traits_type;
	typedef boost::adjacency_list<
		boost::vecS, boost::vecS, boost::directedS,
		boost::property<boost::vertex_name_t, std::string,
			boost::property<boost::vertex_index_t, long,
				boost::property<boost::vertex_color_t, boost::default_color_type,
					boost::property<boost::vertex_distance_t, long,
						boost::property<boost::vertex_predecessor_t, traits_type::edge_descriptor> > > > >,
		boost::property<boost::edge_capacity_t, long,
			boost::property<boost::edge_residual_capacity_t, long,
				boost::property<boost::edge_reverse_t, traits_type::edge_descriptor> > >
	> graph_type;

	graph_type g;
	boost::property_map<graph_type, boost::edge_capacity_t>::type capacity = boost::get(boost::edge_capacity, g);
	boost::property_map<graph_type, boost::edge_residual_capacity_t>::type residual_capacity = boost::get(boost::edge_residual_capacity, g);
	boost::property_map<graph_type, boost::edge_reverse_t>::type reverse_edge = boost::get(boost::edge_reverse, g);
	traits_type::vertex_descriptor src, sink;
	boost::read_dimacs_max_flow(g, capacity, reverse_edge, src, sink, stream);

	std::vector<boost::default_color_type> color(boost::num_vertices(g));
	std::vector<long> distance(boost::num_vertices(g));
	const long flow = boost::boykov_kolmogorov_max_flow(g, src, sink);

	std::cout << "c  The total flow:" << std::endl;
	std::cout << "s " << flow << std::endl << std::endl;

	std::cout << "c flow values:" << std::endl;
	boost::graph_traits<graph_type>::vertex_iterator u_iter, u_end;
	boost::graph_traits<graph_type>::out_edge_iterator ei, e_end;
	for (boost::tie(u_iter, u_end) = boost::vertices(g); u_iter != u_end; ++u_iter)
		for (boost::tie(ei, e_end) = boost::out_edges(*u_iter, g); ei != e_end; ++ei)
			if (capacity[*ei] > 0)
				std::cout << "f " << *u_iter << " " << boost::target(*ei, g) << " " << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
}

// [ref] ${BOOST_HOME}/libs/graph/example/matching_example.cpp
void edmonds_maximum_cardinality_matching_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

	// create the following graph: (it'll look better when output to the terminal in a fixed width font...)
	const int num_vertices = 18;

	std::vector<std::string> ascii_graph;
	ascii_graph.push_back("           0       1---2       3       ");
	ascii_graph.push_back("            \\     /     \\     /        ");
	ascii_graph.push_back("             4---5       6---7         ");
	ascii_graph.push_back("             |   |       |   |         ");
	ascii_graph.push_back("             8---9      10---11        ");
	ascii_graph.push_back("            /     \\     /     \\        ");
	ascii_graph.push_back("     12   13      14---15      16   17 ");

	// it has a perfect matching of size 8. there are two isolated vertices that we'll use later...
	graph_type g(num_vertices);

	// our vertices are stored in a vector, so we can refer to vertices by integers in the range 0..15
	boost::add_edge(0, 4, g);
	boost::add_edge(1, 5, g);
	boost::add_edge(2, 6, g);
	boost::add_edge(3, 7, g);
	boost::add_edge(4, 5, g);
	boost::add_edge(6, 7, g);
	boost::add_edge(4, 8, g);
	boost::add_edge(5, 9, g);
	boost::add_edge(6, 10, g);
	boost::add_edge(7, 11, g);
	boost::add_edge(8, 9, g);
	boost::add_edge(10, 11, g);
	boost::add_edge(8, 13, g);
	boost::add_edge(9, 14, g);
	boost::add_edge(10, 15, g);
	boost::add_edge(11, 16, g);
	boost::add_edge(14, 15, g);

	std::vector<boost::graph_traits<graph_type>::vertex_descriptor> mate(num_vertices);

	// find the maximum cardinality matching.
	// we'll use a checked version of the algorithm, which takes a little longer than the unchecked version,
	// but has the advantage that it will return "false" if the matching returned is not actually a maximum cardinality matching in the graph.
#if 0
	boost::edmonds_maximum_cardinality_matching(g, &mate[0]);
#else
	const bool success1 = checked_edmonds_maximum_cardinality_matching(g, &mate[0]);
	assert(success1);
#endif

	std::cout << "in the following graph:" << std::endl << std::endl;
	for (std::vector<std::string>::iterator itr = ascii_graph.begin(); itr != ascii_graph.end(); ++itr)
		std::cout << *itr << std::endl;

	std::cout << std::endl << "found a matching of size " << boost::matching_size(g, &mate[0]) << std::endl;
	std::cout << "the matching is:" << std::endl;
	boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
	for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		if (mate[*vi] != boost::graph_traits<graph_type>::null_vertex() && *vi < mate[*vi])
			std::cout << "{" << *vi << ", " << mate[*vi] << "}" << std::endl;
	std::cout << std::endl;

	// now we'll add two edges, and the perfect matching has size 9
	ascii_graph.pop_back();
	ascii_graph.push_back("     12---13      14---15      16---17 ");

	boost::add_edge(12, 13, g);
	boost::add_edge(16, 17, g);

#if 0
	boost::edmonds_maximum_cardinality_matching(g, &mate[0]);
#else
	const bool success2 = boost::checked_edmonds_maximum_cardinality_matching(g, &mate[0]);
	assert(success2);
#endif

	std::cout << "in the following graph:" << std::endl << std::endl;
	for (std::vector<std::string>::iterator itr = ascii_graph.begin(); itr != ascii_graph.end(); ++itr)
		std::cout << *itr << std::endl;

	std::cout << std::endl << "found a matching of size " << boost::matching_size(g, &mate[0]) << std::endl;
	std::cout << "the matching is:" << std::endl;
	for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		if (mate[*vi] != boost::graph_traits<graph_type>::null_vertex() && *vi < mate[*vi])
			std::cout << "{" << *vi << ", " << mate[*vi] << "}" << std::endl;
}

void maximum_flow_and_matching()
{
	const std::string max_flow_dat_file("./data/boost/max_flow.dat");
#if defined(__GNUC__)
	std::ifstream stream(max_flow_dat_file.c_str());
#else
	std::ifstream stream(max_flow_dat_file);
#endif

	std::cout << "max-flow algorithm -------------------------------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) max_flow_example(stream);

	std::cout << "\npush–relabel maximum flow algorithm --------------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) push_relabel_example(stream);

	std::cout << "\nBoykov-Kolmogorov (BK) max-flow algorithm --------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) boykov_kolmogorov_example(stream);

	std::cout << "\nEdmonds maximum cardinality matching -------------------------" << std::endl;
	edmonds_maximum_cardinality_matching_example();
}

struct edge_t
{
	unsigned long first;
	unsigned long second;
};

void minimum_cut()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > undirected_graph_type;
	typedef boost::graph_traits<undirected_graph_type>::vertex_descriptor vertex_descriptor_type;
	typedef boost::property_map<undirected_graph_type, boost::edge_weight_t>::type weight_map_type;
	typedef boost::property_traits<weight_map_type>::value_type weight_type;

	// define the 16 edges of the graph. {3, 4} means an undirected edge between vertices 3 and 4.
	const edge_t edges[] = {
		{3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
		{0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}
	};

	// for each of the 16 edges, define the associated edge weight.
	// ws[i] is the weight for the edge that is described by edges[i].
	const weight_type ws[] = { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4 };

	// construct the graph object.
	// 8 is the number of vertices, which are numbered from 0 // through 7, and 16 is the number of edges.
	undirected_graph_type g(edges, edges + 16, ws, 8, 16);

	// define a property map, 'parities', that will store a boolean value for each vertex.
	// vertices that have the same parity after 'stoer_wagner_min_cut' runs are on the same side of the min-cut.
	BOOST_AUTO(parities, boost::make_one_bit_color_map(boost::num_vertices(g), boost::get(boost::vertex_index, g)));

	// run the Stoer-Wagner algorithm to obtain the min-cut weight. `parities` is also filled in.
	const int w = boost::stoer_wagner_min_cut(g, boost::get(boost::edge_weight, g), boost::parity_map(parities));

	std::cout << "the min-cut weight of G is " << w << ".\n" << std::endl;
	assert(w == 7);

	std::cout << "one set of vertices consists of:" << std::endl;
	std::size_t i;
	for (i = 0; i < boost::num_vertices(g); ++i)
	{
		if (boost::get(parities, i))
			std::cout << i << std::endl;
	}
	std::cout << std::endl;

	std::cout << "the other set of vertices consists of:" << std::endl;
	for (i = 0; i < boost::num_vertices(g); ++i)
	{
		if (!boost::get(parities, i))
			std::cout << i << std::endl;
	}
	std::cout << std::endl;
}

void sparse_matrix_ordering()
{
	throw std::runtime_error("not yet implemented");
}

void layout_algorithms()
{
	throw std::runtime_error("not yet implemented");
}

void clustering()
{
	throw std::runtime_error("not yet implemented");
}

void planar_graph_algorithms()
{
	throw std::runtime_error("not yet implemented");
}

void graph_metrics()
{
	throw std::runtime_error("not yet implemented");
}

void graph_structure_comparisons()
{
	throw std::runtime_error("not yet implemented");
}

void graphviz()
{
	// read graphviz
	{
		// vertex properties
		typedef boost::property<boost::vertex_name_t, std::string, boost::property<boost::vertex_color_t, float> > vertex_p;
		// edge properties
		typedef boost::property<boost::edge_weight_t, double> edge_p;
		// graph properties
		typedef boost::property<boost::graph_name_t, std::string> graph_p;
		// adjacency_list-based type
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_p, edge_p, graph_p> graph_type;

		// construct an empty graph and prepare the dynamic_property_maps.
		graph_type graph(0);
		boost::dynamic_properties dp;

		boost::property_map<graph_type, boost::vertex_name_t>::type name = boost::get(boost::vertex_name, graph);
		dp.property("node_id", name);

		boost::property_map<graph_type, boost::vertex_color_t>::type mass = boost::get(boost::vertex_color, graph);
		dp.property("mass", mass);

		boost::property_map<graph_type, boost::edge_weight_t>::type weight = boost::get(boost::edge_weight, graph);
		dp.property("weight", weight);

		// use ref_property_map to turn a graph property into a property map
		boost::ref_property_map<graph_type *, std::string> gname(boost::get_property(graph, boost::graph_name));
		dp.property("name", gname);

		// sample graph as an std::istream;
		std::istringstream gvgraph_stream("digraph { graph [name=\"graphname\"]  a  c e [mass = 6.66] }");

		const bool status = boost::read_graphviz(gvgraph_stream, graph, dp, "node_id");
	}

	// write graphviz
	{
	}
}

}  // local
}  // unnamed namespace

void graph()
{
	std::cout << "basic operation ----------------------------------------------" << std::endl;
	//local::boost_quick_tour();
	local::basic_operation();

	std::cout << "\ntraversal algorithms ----------------------------------------------------" << std::endl;
	local::traversal();

	std::cout << "\nshortest paths / cost minimization algorithms ----------------" << std::endl;
	local::shortest_paths();

	std::cout << "\nother core algorithms ----------------------------------------" << std::endl;
	//local::other_core_algorithms();  // not yet implemented.

	std::cout << "\nminimum spanning tree algorithms -----------------------------" << std::endl;
	local::minimum_spanning_tree();

	std::cout << "\nrandom spanning tree algorithms ------------------------------" << std::endl;
	//local::random_spanning_tree();  // not yet implemented.

	std::cout << "\nalgorithm for common spanning trees of two graphs ------------" << std::endl;
	//local::common_spanning_tree();  // not yet implemented.

	std::cout << "\nconnected components algorithms ------------------------------" << std::endl;
	local::connected_components();

	std::cout << "\nmaximum flow and matching algorithms -------------------------" << std::endl;
	local::maximum_flow_and_matching();

	std::cout << "\nminimum cut algorithms ---------------------------------------" << std::endl;
	local::minimum_cut();

	std::cout << "\nsparse matrix ordering algorithms ----------------------------" << std::endl;
	//local::sparse_matrix_ordering();  // not yet implemented.

	std::cout << "\nlayout algorithms --------------------------------------------" << std::endl;
	//local::layout_algorithms();  // not yet implemented.

	std::cout << "\nclustering algorithms ----------------------------------------" << std::endl;
	//local::clustering();  // not yet implemented.

	std::cout << "\nplanar graph algorithms --------------------------------------" << std::endl;
	//local::planar_graph_algorithms();  // not yet implemented.

	std::cout << "\ngraph metrics ------------------------------------------------" << std::endl;
	//local::graph_metrics();  // not yet implemented.

	std::cout << "\ngraph structure comparisons ----------------------------------" << std::endl;
	//local::graph_structure_comparisons();  // not yet implemented.

	std::cout << "\ngraphviz -----------------------------------------------------" << std::endl;
	local::graphviz();
}
