#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/bellman_ford_shortest_paths.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/astar_search.hpp>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${BOOST_HOME}/libs/graph/example/dijkstra-example.cpp
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
	// VC++ has trouble with the named parameters mechanism.
	boost::property_map<graph_type, boost::vertex_index_t>::type indexmap = boost::get(boost::vertex_index, g);
	boost::dijkstra_shortest_paths(
		g, s, &p[0], &d[0], weightmap, indexmap,
		std::less<int>(), boost::closed_plus<int>(),
		(std::numeric_limits<int>::max)(), 0,
		boost::default_dijkstra_visitor()
	);
#else
	//boost::dijkstra_shortest_paths(g, s, boost::predecessor_map(&p[0]).distance_map(&d[0]));
	boost::dijkstra_shortest_paths(g, s, boost::predecessor_map(boost::make_iterator_property_map(p.begin(), boost::get(boost::vertex_index, g))).distance_map(boost::make_iterator_property_map(d.begin(), boost::get(boost::vertex_index, g))));
#endif

	std::cout << "Distances and parents: " << std::endl;
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

// REF [site] >> https://stackoverflow.com/questions/31145082/bgl-dijkstra-shortest-paths-with-bundled-properties
void dijkstra_bundled_property_example()
{
	struct vertex_properties_t
	{
		std::string label;
		int p1;
		size_t id;
	};

	struct edge_properties_t
	{
		std::string label;
		int p1;
		int weight;
	};

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_properties_t, edge_properties_t> graph_t;
	typedef graph_t::vertex_descriptor vertex_descriptor_t;
	typedef graph_t::edge_descriptor edge_descriptor_t;
	//typedef boost::property_map<graph_t, boost::vertex_index_t>::type index_map_t;
	//typedef boost::iterator_property_map<vertex_descriptor_t*, index_map_t*, vertex_descriptor_t, vertex_descriptor_t&> predecessor_map_t;

	// Create a graph object.
	graph_t g;

	// Add vertices.
	const vertex_descriptor_t v1 = boost::add_vertex(g);
	const vertex_descriptor_t v2 = boost::add_vertex(g);
	const vertex_descriptor_t v3 = boost::add_vertex(g);
	const vertex_descriptor_t v4 = boost::add_vertex(g);

	// Set vertex properties.
	g[v1].p1 = 1;  g[v1].label = "v1";  g[v1].id = 0;
	g[v2].p1 = 2;  g[v2].label = "v2";  g[v2].id = 1;
	g[v3].p1 = 3;  g[v3].label = "v3";  g[v3].id = 2;
	g[v4].p1 = 4;  g[v4].label = "v4";  g[v4].id = 3;

	// Add edges.
	const std::pair<edge_descriptor_t, bool> e01 = boost::add_edge(v1, v2, g);
	const std::pair<edge_descriptor_t, bool> e02 = boost::add_edge(v2, v3, g);
	const std::pair<edge_descriptor_t, bool> e03 = boost::add_edge(v3, v4, g);
	const std::pair<edge_descriptor_t, bool> e04 = boost::add_edge(v4, v1, g);
	const std::pair<edge_descriptor_t, bool> e05 = boost::add_edge(v1, v3, g);

	// Set edge properties.
	g[e01.first].p1 = 1;  g[e01.first].weight = 1;  g[e01.first].label = "v1-v2";
	g[e02.first].p1 = 2;  g[e02.first].weight = 2;  g[e02.first].label = "v2-v3";
	g[e03.first].p1 = 3;  g[e03.first].weight = 1;  g[e03.first].label = "v3-v4";
	g[e04.first].p1 = 4;  g[e04.first].weight = 1;  g[e04.first].label = "v4-v1";
	g[e05.first].p1 = 5;  g[e05.first].weight = 3;  g[e05.first].label = "v1-v3";

	// Print out some useful information.
	std::cout << "Graph:" << std::endl;
	boost::print_graph(g, boost::get(&vertex_properties_t::label, g));
	std::cout << "num_vertices: " << boost::num_vertices(g) << std::endl;
	std::cout << "num_edges: " << boost::num_edges(g) << std::endl;

	// BGL Dijkstra's shortest paths here.
	std::vector<int> distances(boost::num_vertices(g));
	std::vector<vertex_descriptor_t> predecessors(boost::num_vertices(g));

	boost::dijkstra_shortest_paths(
		g, v1,
		boost::weight_map(boost::get(&edge_properties_t::weight, g))
#if 1
			.distance_map(boost::make_iterator_property_map(distances.begin(), boost::get(boost::vertex_index, g)))
			.predecessor_map(boost::make_iterator_property_map(predecessors.begin(), boost::get(boost::vertex_index, g)))
#else
			.distance_map(boost::make_iterator_property_map(distances.begin(), boost::get(&vertex_properties_t::id, g)))
			.predecessor_map(boost::make_iterator_property_map(predecessors.begin(), boost::get(&vertex_properties_t::id, g)))
#endif
	);

	// Extract the shortest path from v1 to v3.
	typedef std::vector<edge_descriptor_t> path_t;
	path_t path;

	vertex_descriptor_t v = v3;
	for(vertex_descriptor_t u = predecessors[v]; u != v; v = u, u = predecessors[v])
	{
		const std::pair<edge_descriptor_t, bool> edge_pair = boost::edge(u, v, g);
		path.push_back(edge_pair.first);
	}

	std::cout << std::endl;
	std::cout << "Shortest path from v1 to v3:" << std::endl;
#if 1
	for(path_t::reverse_iterator riter = path.rbegin(); riter != path.rend(); ++riter)
	{
		const vertex_descriptor_t u_tmp = boost::source(*riter, g);
		const vertex_descriptor_t v_tmp = boost::target(*riter, g);

		std::cout << "  " << g[u_tmp].label << " -> " << g[v_tmp].label << "  (weight: " << g[*riter].weight << ")" << std::endl;
	}
#else
	auto rit = path.rbegin();
	std::cout << pose_graph[boost::source(*rit, pose_graph)].frame_id;
	for(; rit != path.rend(); ++rit)
		std::cout << " -(w: " << pose_graph[*rit].weight << ")-> " << pose_graph[ boost::target(*rit, pose_graph)].frame_id;
	std::cout << std::endl;
#endif
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

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

void shortest_paths()
{
	// Dijkstra's algorithm -----------------------------------------
	local::dijkstra_example();
	local::dijkstra_bundled_property_example();

	// Bellman-Ford algorithm -----------------------------------------
	// REF [file] >> ${BOOST_HOME}/libs/graph/example/bellman-example.cpp

	// REF [site] >> http://www.boost.org/doc/libs/1_59_0/libs/graph/doc/quick_tour.html
	// REF [site] >> http://library.developer.nokia.com/index.jsp?topic=/S60_5th_Edition_Cpp_Developers_Library/GUID-02F20077-73B5-4A63-85DB-D909E0ADE01C/html/con_graph_quick_tour.html
	{
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;
		typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
		typedef std::pair<int, int> edge_type;

		const int num_nodes = 5;
		const edge_type edges[] = { edge_type(0, 2), edge_type(1, 1), edge_type(1, 3), edge_type(1, 4), edge_type(2, 1), edge_type(2, 3), edge_type(3, 4), edge_type(4, 0), edge_type(4, 1) };
		const int edge_weights[] = { 1, 2, 1, 2, 7, 3, 1, 1, 1 };

		graph_type g(edges, edges + sizeof(edges) / sizeof(edge_type), edge_weights, num_nodes);

		// Vector for storing distance property.
		std::vector<int> d(boost::num_vertices(g));
		// Get the first vertex.
		vertex_descriptor_type s = *(boost::vertices(g).first);

		// Invoke variant 2 of Dijkstra's algorithm.
		//boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]));
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(boost::make_iterator_property_map(d.begin(), boost::get(boost::vertex_index, g))));

		std::cout << "distances from start vertex:" << std::endl;
		boost::graph_traits<graph_type>::vertex_iterator vi;
		for (vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
			std::cout << "distance(" << *vi << ") = " << d[*vi] << std::endl;
		std::cout << std::endl;

		//
		std::vector<vertex_descriptor_type> p(boost::num_vertices(g), boost::graph_traits<graph_type>::null_vertex());  // The predecessor array.
		//boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]).visitor(make_predecessor_recorder(&p[0])));
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(boost::make_iterator_property_map(d.begin(), boost::get(boost::vertex_index, g))).visitor(local::make_predecessor_recorder(&p[0])));

		std::cout << "Parents in the tree of shortest paths:" << std::endl;
		for (vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
		{
			std::cout << "parent(" << *vi;
			//if (p[*vi] == vertex_descriptor_type() && *vi == s)
			if (p[*vi] == boost::graph_traits<graph_type>::null_vertex())  // Not working.
				std::cout << ") = no parent" << std::endl;
			else
				std::cout << ") = " << p[*vi] << std::endl;
		}
	}
}

}  // namespace boost_graph
