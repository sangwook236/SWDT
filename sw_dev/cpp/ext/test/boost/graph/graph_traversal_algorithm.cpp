#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/astar_search.hpp>
#include <iostream>


namespace {
namespace local {

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

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

// REF [site] >> http://www.ibm.com/developerworks/aix/library/au-aix-boost-graph/index.html
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

		local::custom_dfs_visitor vis;
		boost::depth_first_search(g, boost::visitor(vis));
	}


	std::cout << "\nbreadth-first search -----------------------------------------" << std::endl;
	{
        // REF [file] >> ${BOOST_HOME}/libs/graph/example/bfs.cpp
	}

	std::cout << "\nA* search ----------------------------------------------------" << std::endl;
	{
        // REF [file] >> ${BOOST_HOME}/libs/graph/example/astar-cities.cpp
        // REF [file] >> ${BOOST_HOME}/libs/graph/example/astar_maze.cpp
	}
}

}  // namespace boost_graph
