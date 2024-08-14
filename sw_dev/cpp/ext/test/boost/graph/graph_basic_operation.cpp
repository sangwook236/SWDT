#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/grid_graph.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/utility.hpp>
#include <boost/foreach.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <string>
#include <utility>


//#define __USE_BOOST_PROPERTY_KIND 1

#if defined(__USE_BOOST_PROPERTY_KIND)
namespace boost {

enum my_vertex_population_type { my_vertex_population };
enum my_vertex_zipcodes_type { my_vertex_zipcodes };
template <>
struct property_kind<my_vertex_population_type>
{
	typedef vertex_property_tag type;
};

enum my_edge_speed_limit_type { my_edge_speed_limit };
enum my_edge_lanes_type { my_edge_lanes };
enum my_edge_divided_type { my_edge_divided };
template <>
struct property_kind<my_edge_speed_limit_type>
{
	typedef edge_property_tag type;
};

enum my_graph_use_right_type { my_graph_use_right };
enum my_graph_use_metric_type { my_graph_use_metric };

}
#endif

namespace boost_graph {

template <class Graph>
struct exercise_vertex
{
	exercise_vertex(Graph &g, const char *name)
	: g_(g), name_(name)
	{}

	typedef typename boost::graph_traits<Graph>::vertex_descriptor vertex_descriptor_type;

	void operator()(const vertex_descriptor_type &v) const
	{
		typename boost::property_map<Graph, boost::vertex_index_t>::type vertex_ids = boost::get(boost::vertex_index, g_);

		std::cout << "Vertex: " << name_[boost::get(vertex_ids, v)] << std::endl;

		// Write out the outgoing edges.
		std::cout << "\tOut-edges: ";
		typename boost::graph_traits<Graph>::out_edge_iterator out_i, out_end;
		typename boost::graph_traits<Graph>::edge_descriptor e;
		for (boost::tie(out_i, out_end) = boost::out_edges(v, g_); out_i != out_end; ++out_i)
		{
			e = *out_i;
			vertex_descriptor_type src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_ids, src)] << "," << name_[boost::get(vertex_ids, targ)] << ") ";
		}
		std::cout << std::endl;

		// Write out the incoming edges.
		std::cout << "\tIn-edges: ";
		typename boost::graph_traits<Graph>::in_edge_iterator in_i, in_end;
		// NOTE [info] >> Directed graphs can only store outgoing edges.
		for (boost::tie(in_i, in_end) = boost::in_edges(v, g_); in_i != in_end; ++in_i)
		{
			e = *in_i;
			vertex_descriptor_type src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_ids, src)] << "," << name_[boost::get(vertex_ids, targ)] << ") ";
		}
		std::cout << std::endl;

		// Write out all adjacent vertices.
		std::cout << "\tAdjacent vertices: ";
		typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
		for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, g_); ai != ai_end; ++ai)
			std::cout << name_[boost::get(vertex_ids, *ai)] <<  " ";
		std::cout << std::endl;
	}

private:
	Graph &g_;
	const char *name_;
};

template <typename Graph>
struct EdgeComparator
{
public:
	EdgeComparator(const size_t numVertices)
	: numVertices_(numVertices)
	{}

	bool operator()(const typename Graph::edge_descriptor& lhs, const typename Graph::edge_descriptor& rhs) const
	{
		const size_t lsrc = lhs.m_source < lhs.m_target ? lhs.m_source : lhs.m_target, ldst = lhs.m_source < lhs.m_target ? lhs.m_target : lhs.m_source;
		const size_t rsrc = rhs.m_source < rhs.m_target ? rhs.m_source : rhs.m_target, rdst = rhs.m_source < rhs.m_target ? rhs.m_target : rhs.m_source;
		//return lsrc < rsrc || ldst < rdst;
		return (lsrc * numVertices_ + ldst) < (rsrc * numVertices_ + rdst);
	}

private:
	const size_t numVertices_;
};

// REF [file] >> ${BOOST_HOME}/libs/graph/example/quick_tour.cpp
void boost_quick_tour()
{
	// Create a typedef for the graph_type type.
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_weight_t, float> > graph_type;

	// Make convenient labels for the vertices.
	enum { A, B, C, D, E, N };
	const int num_vertices = N;
	const char name[] = "ABCDE";

	// Writing out the edges in the graph.
	typedef std::pair<int, int> edge_type;

	const edge_type edge_array[] = { edge_type(A, B), edge_type(A, D), edge_type(C, A), edge_type(D, C), edge_type(C, E), edge_type(B, D), edge_type(D, E), };
	const int num_edges = sizeof(edge_array) / sizeof(edge_array[0]);

	// Average transmission delay (in milliseconds) for each connection.
	const float transmission_delay[] = { 1.2f, 4.5f, 2.6f, 0.4f, 5.2f, 1.8f, 3.3f, 9.1f };

	// Declare a graph object, adding the edges and edge properties.
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// VC++ can't handle the iterator constructor.
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

	// Can use boost::vertex_index_t even though we didn't define any property for vertices (boost::no_property).
	boost::property_map<graph_type, boost::vertex_index_t>::type vertex_ids = boost::get(boost::vertex_index, g);
	boost::property_map<graph_type, boost::edge_weight_t>::type trans_delays = boost::get(boost::edge_weight, g);

	// Access vertices and their properties.
	std::cout << "boost::vertices(g) = ";
	typedef boost::graph_traits<graph_type>::vertex_iterator vertex_iterator_type;
	std::pair<vertex_iterator_type, vertex_iterator_type> vp;
	for (vp = boost::vertices(g); vp.first != vp.second; ++vp.first)
	{
		std::cout << name[boost::get(vertex_ids, *vp.first)] <<  " ";
		//std::cout << name[vertex_ids[*vp.first]] <<  " ";
	}
	std::cout << std::endl;

	// Access edges and their properties.
	std::cout << "boost::edges(g) = ";
	boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
	{
		std::cout << "(" << name[boost::get(vertex_ids, boost::source(*ei, g))] << "," << name[boost::get(vertex_ids, boost::target(*ei, g))] << ") ";
		//std::cout << "(" << name[vertex_ids[boost::source(*ei, g)]] << "," << name[vertex_ids[boost::target(*ei, g)]] << ") ";
	}
	std::cout << std::endl;

	//
	std::for_each(boost::vertices(g).first, boost::vertices(g).second, exercise_vertex<graph_type>(g, name));

	const auto &[eit_begin, eit_end] = boost::edges(g);
	const auto eit = std::find_if(eit_begin, eit_end, [&weightmap](const auto &elem) { return weightmap[elem] > 2.0f; });
	std::cout << *eit << std::endl;

	// Output.
	std::map<std::string, std::string> graph_attr, vertex_attr, edge_attr;
	graph_attr["size"] = "3,3";
	graph_attr["rankdir"] = "LR";
	graph_attr["ratio"] = "fill";
	vertex_attr["shape"] = "circle";

	boost::write_graphviz(
		std::cout,
		g,
		boost::make_label_writer(name),  // Vertex properties writer.
		boost::make_label_writer(trans_delays),  // Edge properties writer.
		boost::make_graph_attributes_writer(graph_attr, vertex_attr, edge_attr)  // Graph properties writer.
	);
}

// REF [site] >> http://www.ibm.com/developerworks/aix/library/au-aix-boost-graph/index.html
// REF [site] >> Example 31.4 in http://www.theboostcpplibraries.com/boost.graph-vertices-and-edges
void basic_operation_1()
{
	std::cout << "--------------------------------------------------------------" << std::endl;
	{
		// The Directed template parameter controls whether the graph is directed, undirected, or directed with access to both the in-edges and out-edges (bidirectional).
#if 0
		// Use boost::listS as OutEdgeListS.
		// Multiple edges between two vertices are possible.
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> undirected_graph_type;
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS> directed_graph_type;
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS> bidirectional_graph_type;
#else
		// boost::setS is used as the first template parameter (OutEdgeListS).
		// Only one edge between two vertices can exist.
		typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS> undirected_graph_type;
		typedef boost::adjacency_list<boost::setS, boost::vecS, boost::directedS> directed_graph_type;
		typedef boost::adjacency_list<boost::setS, boost::vecS, boost::bidirectionalS> bidirectional_graph_type;
#endif

		undirected_graph_type ungraph(4);
		boost::add_edge(0, 1, ungraph);
		//boost::add_edge(1, 0, ungraph);
		std::cout << "Is a duplicate edge added to undirected graph? " << std::boolalpha << boost::add_edge(0, 1, ungraph).second << std::endl;
		std::cout << "Is a duplicate edge added to undirected graph? " << std::boolalpha << boost::add_edge(1, 0, ungraph).second << std::endl;
		boost::add_edge(0, 2, ungraph);
		boost::add_edge(3, 0, ungraph);
		//boost::add_edge(0, 0, ungraph);

		directed_graph_type digraph(4);
		boost::add_edge(0, 1, digraph);
		//boost::add_edge(1, 0, digraph);
		std::cout << "Is a duplicate edge added to directed graph? " << std::boolalpha << boost::add_edge(0, 1, digraph).second << std::endl;
		std::cout << "Is a duplicate edge added to directed graph? " << std::boolalpha << boost::add_edge(1, 0, digraph).second << std::endl;
		boost::add_edge(0, 2, digraph);
		boost::add_edge(3, 0, digraph);
		//boost::add_edge(0, 0, digraph);

		bidirectional_graph_type bigraph(4);
		boost::add_edge(0, 1, bigraph);
		//boost::add_edge(1, 0, bigraph);
		std::cout << "Is a duplicate edge added to bidirectional graph? " << std::boolalpha << boost::add_edge(0, 1, bigraph).second << std::endl;
		std::cout << "Is a duplicate edge added to bidirectional graph? " << std::boolalpha << boost::add_edge(1, 0, bigraph).second << std::endl;
		boost::add_edge(0, 2, bigraph);
		boost::add_edge(3, 0, bigraph);
		//boost::add_edge(0, 0, bigraph);

		std::cout << "Number of edges in the undirected graph = " << boost::num_edges(ungraph) << std::endl;
		std::cout << "Number of edges in the directed graph = " << boost::num_edges(digraph) << std::endl;
		std::cout << "Number of edges in the bidirectional graph = " << boost::num_edges(bigraph) << std::endl;

		std::pair<undirected_graph_type::edge_iterator, undirected_graph_type::edge_iterator> es1 = boost::edges(ungraph);
		std::copy(es1.first, es1.second, std::ostream_iterator<undirected_graph_type::edge_descriptor>(std::cout, ", "));
		std::cout << std::endl;
		std::pair<directed_graph_type::edge_iterator, directed_graph_type::edge_iterator> es2 = boost::edges(digraph);
		std::copy(es2.first, es2.second, std::ostream_iterator<directed_graph_type::edge_descriptor>(std::cout, ", "));
		std::cout << std::endl;
		std::pair<bidirectional_graph_type::edge_iterator, bidirectional_graph_type::edge_iterator> es3 = boost::edges(bigraph);
		std::copy(es3.first, es3.second, std::ostream_iterator<bidirectional_graph_type::edge_descriptor>(std::cout, ", "));
		std::cout << std::endl;

		//const auto &graph = ungraph;
		//const auto &graph = digraph;  // Error.
		const auto &graph = bigraph;
		const auto &[avit_begin, avit_end] = boost::adjacent_vertices(0, graph);  // Vertices of outgoing edges.
		std::cout << "Vertices of outgoing edges: ";
		std::copy(avit_begin, avit_end, std::ostream_iterator<size_t>(std::cout, ", "));
		std::cout << std::endl;
		const auto &[iavit_begin, iavit_end] = boost::inv_adjacent_vertices(0, graph);  // Vertices of incoming edges.
		std::cout << "Vertices of incoming edges: ";
		std::copy(iavit_begin, iavit_end, std::ostream_iterator<size_t>(std::cout, ", "));
		std::cout << std::endl;
	}

	std::cout << "\n--------------------------------------------------------------" << std::endl;
	{
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS> graph_type;

		// Create a simple undirected graph.
		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(0, 3, g);
		boost::add_edge(1, 2, g);
		boost::add_edge(2, 3, g);

		graph_type::vertex_iterator vertexIt, vertexEnd;  // Iterate over all the vertices of the graph.
		graph_type::adjacency_iterator neighborIt, neighborEnd;  // Iterate over the corresponding adjacent vertices.
		graph_type::out_edge_iterator edgeIt, edgeEnd;  // Iterate over all the incident edges of a vertex.
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			// Access adjacent vertices of each vertex.
			std::cout << *vertexIt << " is connected with ";
			boost::tie(neighborIt, neighborEnd) = boost::adjacent_vertices(*vertexIt, g);
			for (; neighborIt != neighborEnd; ++neighborIt)
				std::cout << *neighborIt << " ";
			std::cout << std::endl;

			// The number of adjacent vertices of a vertex.
			// boost::degree(), boost::in_degree(), and boost::out_degree() can be used instead.

			// Access incident edges of each vertex.
			// REF [function] >> default_undirected_and_directed_graph().
			std::cout << *vertexIt << " is incident to ";
			boost::tie(edgeIt, edgeEnd) = boost::incident_edges(*vertexIt, g);
			for (; edgeIt != edgeEnd; ++edgeIt)
				std::cout << *edgeIt << " ";
			std::cout << std::endl;
		}
	}

	std::cout << "\n--------------------------------------------------------------" << std::endl;
	{
		// When using the directedS tag in BGL, you are allowed to use only the out_edges helper function and associated iterators.
		// Using in_edges requires changing the graph type to bidirectionalS, although this is still more or less a directed graph.
		//typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS> graph_type;
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS> graph_type;

		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(0, 3, g);
		boost::add_edge(1, 2, g);
		boost::add_edge(2, 3, g);

		graph_type::vertex_iterator vertexIt, vertexEnd;
		graph_type::in_edge_iterator inEdgeIt, inEdgeEnd;

		// Access incoming edges of each vertex.
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "In-edges for " << *vertexIt << ": ";
			// NOTE [info] >> Directed graphs can only store outgoing edges.
			boost::tie(inEdgeIt, inEdgeEnd) = boost::in_edges(*vertexIt, g);
			for (; inEdgeIt != inEdgeEnd; ++inEdgeIt)
				std::cout << *inEdgeIt << " ";
			std::cout << std::endl;
		}

		//
		graph_type::out_edge_iterator outEdgeIt, outEdgeEnd;

		// Access outgoing edges of each vertex.
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "Out-edges for " << *vertexIt << ": ";
			boost::tie(outEdgeIt, outEdgeEnd) = boost::out_edges(*vertexIt, g);  // Similar to incoming edges.
			for (; outEdgeIt != outEdgeEnd; ++outEdgeIt)
				std::cout << *outEdgeIt << " ";
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

		std::cout << "Number of edges: " << boost::num_edges(g) << std::endl;
		std::cout << "Number of vertices: " << boost::num_vertices(g) << std::endl;

		graph_type::vertex_iterator vertexIt, vertexEnd;
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			// NOTE [info] >> Directed graphs can only store outgoing edges.
			std::cout << "In-degree for " << *vertexIt << ": " << boost::in_degree(*vertexIt, g) << std::endl;
			std::cout << "Out-degree for " << *vertexIt << ": " << boost::out_degree(*vertexIt, g) << std::endl;
		}

		graph_type::edge_iterator edgeIt, edgeEnd;
		boost::tie(edgeIt, edgeEnd) = boost::edges(g);
		for (; edgeIt != edgeEnd; ++edgeIt)
			std::cout << "Edge " << boost::source(*edgeIt, g) << "-->" << boost::target(*edgeIt, g) << std::endl;

		// Sort edges.
		//boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
		//boost::tie(ei, ei_end) = boost::edges(g);
		std::pair<graph_type::edge_iterator, graph_type::edge_iterator> eis = boost::edges(g);
		std::list<graph_type::edge_descriptor> edges(eis.first, eis.second);
		edges.sort(EdgeComparator<graph_type>(boost::num_vertices(g)));

		std::cout << "Sorted edges: ";
		std::copy(edges.begin(), edges.end(), std::ostream_iterator<graph_type::edge_descriptor>(std::cout, ", "));
		std::cout << std::endl;
	}
}

void basic_operation_2()
{
	std::cout << "-------------------------------------------------------------" << std::endl;
	{
		using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS>;
		//using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS>;
		//using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

		graph_type g;

		auto v1 = boost::add_vertex(g);
		auto v2 = boost::add_vertex(g);
		auto v3 = boost::add_vertex(g);

		auto e12 = boost::add_edge(v1, v2, g);
		auto e23 = boost::add_edge(v2, v3, g);

		std::cout << "Type of vertex = " << typeid(graph_type::vertex_descriptor).name() << std::endl;  // size_t
		std::cout << "Type of vertex = " << typeid(v1).name() << std::endl;  // size_t
		std::cout << "Vertex 1 = " << v1 << std::endl;
		std::cout << "Vertex 2 = " << v2 << std::endl;
		std::cout << "Vertex 3 = " << v3 << std::endl;
		std::cout << "Type of vertex = " << typeid(graph_type::edge_descriptor).name() << std::endl;
		std::cout << "Type of edge = " << typeid(e12.first).name() << std::endl;
		std::cout << "Edge 12 = " << e12.first << std::endl;
		std::cout << "Edge 23 = " << e23.first << std::endl;

		const auto &vv1 = boost::vertex(0, g);
		const auto &vv2 = boost::vertex(1, g);
		const auto &vv3 = boost::vertex(2, g);

		const auto &ee12 = boost::edge(vv1, vv2, g);
		const auto &ee23 = boost::edge(vv2, vv3, g);

		//std::cout << "Type of vertex = " << typeid(graph_type::vertex_descriptor).name() << std::endl;  // size_t
		std::cout << "Type of vertex = " << typeid(vv1).name() << std::endl;  // size_t
		std::cout << "Vertex 1 = " << vv1 << std::endl;
		std::cout << "Vertex 2 = " << vv2 << std::endl;
		std::cout << "Vertex 3 = " << vv3 << std::endl;
		//std::cout << "Type of vertex = " << typeid(graph_type::edge_descriptor).name() << std::endl;
		std::cout << "Type of edge = " << typeid(ee12.first).name() << std::endl;
		std::cout << "Edge 12 = " << ee12.first << std::endl;
		std::cout << "Edge 23 = " << ee23.first << std::endl;
	}

	std::cout << "-------------------------------------------------------------" << std::endl;
	{
		using graph_type = boost::adjacency_list<boost::setS, boost::setS, boost::bidirectionalS>;
		//using graph_type = boost::adjacency_list<boost::setS, boost::setS, boost::directedS>;
		//using graph_type = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS>;

		graph_type g;

		auto v1 = boost::add_vertex(g);
		auto v2 = boost::add_vertex(g);
		auto v3 = boost::add_vertex(g);

		auto e12 = boost::add_edge(v1, v2, g);
		auto e23 = boost::add_edge(v2, v3, g);

		std::cout << "Type of vertex = " << typeid(graph_type::vertex_descriptor).name() << std::endl;  // void *
		std::cout << "Type of vertex = " << typeid(v1).name() << std::endl;  // void *
		std::cout << "Vertex 1 = " << v1 << std::endl;
		std::cout << "Vertex 2 = " << v2 << std::endl;
		std::cout << "Vertex 3 = " << v3 << std::endl;
		std::cout << "Type of vertex = " << typeid(graph_type::edge_descriptor).name() << std::endl;
		std::cout << "Type of edge = " << typeid(e12.first).name() << std::endl;
		std::cout << "Edge 12 = " << e12.first << std::endl;
		std::cout << "Edge 23 = " << e23.first << std::endl;

		const auto &vv1 = boost::vertex(0, g);
		const auto &vv2 = boost::vertex(1, g);
		const auto &vv3 = boost::vertex(2, g);

		const auto &ee12 = boost::edge(vv1, vv2, g);
		const auto &ee23 = boost::edge(vv2, vv3, g);

		//std::cout << "Type of vertex = " << typeid(graph_type::vertex_descriptor).name() << std::endl;  // void *
		std::cout << "Type of vertex = " << typeid(vv1).name() << std::endl;  // void *
		std::cout << "Vertex 1 = " << vv1 << std::endl;
		std::cout << "Vertex 2 = " << vv2 << std::endl;
		std::cout << "Vertex 3 = " << vv3 << std::endl;
		//std::cout << "Type of vertex = " << typeid(graph_type::edge_descriptor).name() << std::endl;
		std::cout << "Type of edge = " << typeid(ee12.first).name() << std::endl;
		std::cout << "Edge 12 = " << ee12.first << std::endl;
		std::cout << "Edge 23 = " << ee23.first << std::endl;
	}
}

void basic_operation_3()
{
	// Make convenient labels for the vertices.
	enum { A, B, C, D, E, N };
	//enum { A = 10, B, C, D, E, N = 5 };  // Create a graph with 14 vertices (0 ~ 14)
	//enum { A, B, C, D, E = 30, N = 5 };  // Create a graph with 31 vertices (0 ~ 30)
	const int num_vertices = N;

	// Writing out the edges in the graph.
	using edge_type = std::pair<int, int>;

	const edge_type edge_array[] = { edge_type(A, B), edge_type(A, D), edge_type(C, A), edge_type(D, C), edge_type(C, E), edge_type(B, D), edge_type(D, E), };
	const int num_edges = sizeof(edge_array) / sizeof(edge_array[0]);

	const float edge_weights[] = { 1.2f, 4.5f, 2.6f, 0.4f, 5.2f, 1.8f, 3.3f, 9.1f };

	std::cout << "-------------------------------------------------------------" << std::endl;
	{
		// Create a typedef for the graph_type type.
		using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::no_property>;

		// Declare a graph object, adding the edges and edge properties.
#if 1
		graph_type g(num_vertices);
		for (int i = 0; i < num_edges; ++i)
		{
			const edge_type &e = edge_array[i];
			const auto &edge = boost::add_edge(e.first, e.second, g);
			if (edge.second)
			{
				//g[edge.first] = edge_weights[i];  // Compile error
				//boost::get(g, edge.first) = edge_weights[i];  // Compile error
				//boost::put(g, edge.first, edge_weights[i]);  // Compile error
			}
		}
#else
		graph_type g(edge_array, edge_array + num_edges, edge_weights, num_vertices);  // Compile error
#endif

		{
#if 0
			for (int i = 0; i < num_vertices; ++i)
			{
				//g[i] = float(i * 10);  // Compile error
			}
#else
			int i = 0;
			//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
			graph_type::vertex_iterator vi, vi_end;
			for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi, ++i)
			{
				//g[*vi] = float(i * 10);  // Compile error
				//boost::get(g, *vi) = float(i * 10);  // Compile error
				//boost::put(g, *vi, float(i * 10));  // Compile error
			}
#endif
		}
		std::cout << "#vertices = " << boost::num_vertices(g) << ", #edges = " << boost::num_edges(g) << std::endl;

		//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
		graph_type::vertex_iterator vi, vi_end;
		for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		{
			//std::cout << "Vertex " << *vi << ": " << g[*vi] << std::endl;  // Compile error
			//std::cout << "Vertex " << *vi << ": " << boost::get(g, *vi) << std::endl;  // Compile error
		}
		//boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
		graph_type::edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		{
			//std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << g[*ei] << std::endl;  // Compile error
			//std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << boost::get(g, *ei) << std::endl;  // Compile error
		}
	}

	std::cout << "-------------------------------------------------------------" << std::endl;
	{
		// Create a typedef for the graph_type type.
		using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, float, float>;

		// Declare a graph object, adding the edges and edge properties.
#if 1
		graph_type g(num_vertices);
		//boost::property_map<graph_type, float>::type edge_prop_map = boost::get(0.0f, g);  // Compile error
		for (int i = 0; i < num_edges; ++i)
		{
			const edge_type &e = edge_array[i];
			const auto &edge = boost::add_edge(e.first, e.second, g);
			if (edge.second)
			{
				g[edge.first] = edge_weights[i];
				//boost::get(g, edge.first) = edge_weights[i];  // Compile error
				//boost::put(g, edge.first, edge_weights[i]);  // Compile error
			}
		}
#else
		graph_type g(edge_array, edge_array + num_edges, edge_weights, num_vertices);
		//boost::property_map<graph_type, float>::type edge_prop_map = boost::get(0.0f, g);  // Compile error
#endif

		//boost::property_map<graph_type, float>::type vertex_prop_map = boost::get(0.0f, g);  // Compile error
		{
#if 0
			for (int i = 0; i < num_vertices; ++i)
			{
				g[i] = float(i * 10);
			}
#else
			int i = 0;
			//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
			graph_type::vertex_iterator vi, vi_end;
			for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi, ++i)
			{
				g[*vi] = float(i * 10);
				//boost::get(g, *vi) = float(i * 10);  // Compile error
				//boost::put(g, *vi, float(i * 10));  // Compile error
			}
#endif
		}
		std::cout << "#vertices = " << boost::num_vertices(g) << ", #edges = " << boost::num_edges(g) << std::endl;

		//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
		graph_type::vertex_iterator vi, vi_end;
		for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		{
			std::cout << "Vertex " << *vi << ": " << g[*vi] << std::endl;
			//std::cout << "Vertex " << *vi << ": " << boost::get(g, *vi) << std::endl;  // Compile error
		}
		//boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
		graph_type::edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		{
			std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << g[*ei] << std::endl;
			//std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << boost::get(g, *ei) << std::endl;  // Compile error
		}
	}

	std::cout << "-------------------------------------------------------------" << std::endl;
	{
		// Create a typedef for the graph_type type.
		//using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::property<boost::vertex_name_t, std::string>, boost::property<boost::edge_weight_t, float>>;
		using graph_type = boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::property<boost::vertex_attribute_t, float>, boost::property<boost::edge_weight_t, float>>;

		// Declare a graph object, adding the edges and edge properties.
#if 1
		graph_type g(num_vertices);
		boost::property_map<graph_type, boost::edge_weight_t>::type edge_weight_map = boost::get(boost::edge_weight, g);
		for (int i = 0; i < num_edges; ++i)
		{
			const edge_type &e = edge_array[i];
			const auto &edge = boost::add_edge(e.first, e.second, g);
			if (edge.second)
			{
				//edge_weight_map[edge.first] = edge_weights[i];
				//boost::get(edge_weight_map, edge.first) = edge_weights[i];
				boost::put(edge_weight_map, edge.first, edge_weights[i]);
			}
		}
#else
		graph_type g(edge_array, edge_array + num_edges, edge_weights, num_vertices);
		boost::property_map<graph_type, boost::edge_weight_t>::type edge_weight_map = boost::get(boost::edge_weight, g);
#endif

		//boost::property_map<graph_type, boost::vertex_name_t>::type vertex_name_map = boost::get(boost::vertex_name, g);
		boost::property_map<graph_type, boost::vertex_attribute_t>::type vertex_attribute_map = boost::get(boost::vertex_attribute, g);
		{
#if 0
			for (int i = 0; i < num_vertices; ++i)
			{
				//vertex_name_map[i] = std::to_string(i * 10);
				vertex_attribute_map[i] = float(i * 10);
			}
#else
			int i = 0;
			//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
			graph_type::vertex_iterator vi, vi_end;
			for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi, ++i)
			{
				//vertex_name_map[*vi] = std::to_string(i * 10);
				//boost::get(vertex_name_map, *vi) = std::to_string(i * 10);
				//boost::put(vertex_name_map, *vi, std::to_string(i * 10));
				//vertex_attribute_map[*vi] = float(i * 10);
				//boost::get(vertex_attribute_map, *vi) = float(i * 10);
				boost::put(vertex_attribute_map, *vi, float(i * 10));
			}
#endif
		}
		std::cout << "#vertices = " << boost::num_vertices(g) << ", #edges = " << boost::num_edges(g) << std::endl;

		//boost::graph_traits<graph_type>::vertex_iterator vi, vi_end;
		graph_type::vertex_iterator vi, vi_end;
		for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		{
			//std::cout << "Vertex " << *vi << ": " << vertex_name_map[*vi] << std::endl;
			//std::cout << "Vertex " << *vi << ": " << boost::get(vertex_name_map, *vi) << std::endl;
			//std::cout << "Vertex " << *vi << ": " << vertex_attribute_map[*vi] << std::endl;
			std::cout << "Vertex " << *vi << ": " << boost::get(vertex_attribute_map, *vi) << std::endl;
		}
		//boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
		graph_type::edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		{
			//std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << edge_weight_map[*ei] << std::endl;
			std::cout << "Edge " << boost::source(*ei, g) << " -> " << boost::target(*ei, g) << ": " << boost::get(edge_weight_map, *ei) << std::endl;
		}

		std::cout << "-----" << std::endl;
		//boost::print_graph(g, vertex_name_map);  // Better
		boost::print_graph(g, vertex_attribute_map);
	}
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/graph/doc/bundles.html
void bundled_properties_1()
{
	// With bundled properties.
	{
		// Vertex bundle.
		struct City
		{
			std::string name;
			int population;
			std::vector<int> zipcodes;
		};

		// Edge bundle.
		struct Highway
		{
			std::string name;
			double miles;
			int speed_limit;
			int lanes;
			bool divided;
		};

		// Graph bundle.
		struct Country
		{
			std::string name;
			bool use_right;  // Drive on the left or right.
			bool use_metric;  // mph or km/h.
		};

		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS, City, Highway, Country> map_type;

		map_type map(5);
		{
			boost::add_edge(0, 1, map);
			boost::add_edge(1, 2, map);
			boost::add_edge(2, 3, map);
			boost::add_edge(3, 4, map);
			boost::add_edge(0, 2, map);

			// Accessing bundled properties.
			map_type::vertex_descriptor v = *boost::vertices(map).first;  // The 0th vertex.
			map[v].name = "Troy";
			map[v].population = 49170;
			map[v].zipcodes.push_back(12180);
			map_type::edge_descriptor e = *boost::out_edges(v, map).first;  // The 0th outgoing edge of a vertex, v.
			map[e].name = "I-87";
			map[e].miles = 10.;
			map[e].speed_limit = 65;
			map[e].lanes = 4;
			map[e].divided = true;

#if 0
			boost::set_property(map, &Country::name, "Republic of Korea");
			boost::set_property(map, &Country::use_right, true);
			boost::set_property(map, &Country::use_metric, true);
#else
			Country country;
			country.name = "Republic of Korea";
			country.use_right = true;
			country.use_metric = true;
			boost::set_property(map, boost::graph_bundle, country);
#endif

			//auto map_attrs = boost::get_property(map, boost::graph_bundle);
			//map_attrs.name = "Japan";
			//map_attrs.use_metric = true;
			//map_attrs.use_right = false;

			// NOTE [error] >> Compile-time error.
			/*
			map[boost::vertex_bundle].name = "TROY";  // Change the corresponding property of 0th vertex.
			map[boost::vertex_bundle].population = -49170;  // Change the corresponding property of 0th vertex.

			map[boost::edge_bundle].name = "I-78";  // Change the corresponding property of 0th vertex.
			map[boost::edge_bundle].miles = -10.0;
			map[boost::edge_bundle].speed_limit = -65;
			map[boost::edge_bundle].lanes = -4;
			map[boost::edge_bundle].divided = false;
			*/

			map[boost::graph_bundle].name = "United States";
			map[boost::graph_bundle].use_right = true;
			map[boost::graph_bundle].use_metric = false;
		}

		// Output.
		// REF [function] >> bfs_name_printer_example() in graph_traversal_algorithm.cpp.
		boost::property_map<map_type, std::string City::*>::type vertex_names = boost::get(&City::name, map);
		boost::property_map<map_type, int City::*>::type vertex_populations = boost::get(&City::population, map);
		boost::property_map<map_type, std::string Highway::*>::type edge_names = boost::get(&Highway::name, map);
		boost::property_map<map_type, double Highway::*>::type edge_miles = boost::get(&Highway::miles, map);

		boost::write_graphviz(
			std::cout, map,
			boost::make_label_writer(vertex_names),  // Vertex properties writer.
			//boost::make_label_writer(vertex_populations),  // Vertex properties writer.
			boost::make_label_writer(edge_names)  // Edge properties writer.
			//boost::make_label_writer(edge_miles)  // Edge properties writer.
		);
	}

	// Without bundled properties.
	{
#if defined(__USE_BOOST_PROPERTY_KIND)
		typedef boost::adjacency_list<
			boost::listS, boost::vecS, boost::bidirectionalS,
			// Vertex properties.
			boost::property<boost::vertex_name_t, std::string,
				boost::property<boost::my_vertex_population_type, int,
					boost::property<boost::my_vertex_zipcodes_type, std::vector<int> >
				>
			>,
			// Edge properties.
			boost::property<boost::edge_name_t, std::string,
				boost::property<boost::edge_weight_t, double,
					boost::property<boost::my_edge_speed_limit_type, int,
						boost::property<boost::my_edge_lanes_type, int,
							boost::property<boost::my_edge_divided_type, bool>
						>
					>
				>
			>,
			// Graph properties.
			boost::property<boost::graph_name_t, std::string,
				boost::property<boost::my_graph_use_right_type, bool,
					boost::property<boost::my_graph_use_metric_type, bool>
				>
			>
		> map_type;
#else
		// REF [struct] >> struct edge_component_t in graph.cpp.
		struct vertex_population_type
		{
			enum { num = 601 };
			typedef boost::vertex_property_tag kind;
		} vertex_population;
		struct vertex_zipcodes_type
		{
			enum { num = 602 };
			typedef boost::vertex_property_tag kind;
		} vertex_zipcodes;
		struct edge_speed_limit_type
		{
			enum { num = 701 };
			typedef boost::edge_property_tag kind;
		} edge_speed_limit;
		struct edge_lanes_type
		{
			enum { num = 702 };
			typedef boost::edge_property_tag kind;
		} edge_lanes;
		struct edge_divided_type
		{
			enum { num = 703 };
			typedef boost::edge_property_tag kind;
		} edge_divided;
		struct graph_use_right_type
		{
			enum { num = 801 };
			typedef boost::graph_property_tag kind;
		} graph_use_right;
		struct graph_use_metric_type
		{
			enum { num = 802 };
			typedef boost::graph_property_tag kind;
		} graph_use_metric;

		typedef boost::adjacency_list<
			boost::listS, boost::vecS, boost::bidirectionalS,
			// Vertex properties.
			boost::property<boost::vertex_name_t, std::string,
				boost::property<vertex_population_type, int,
					boost::property<vertex_zipcodes_type, std::vector<int> >
				>
			>,
			// Edge properties.
			boost::property<boost::edge_name_t, std::string,
				boost::property<boost::edge_weight_t, double,
					boost::property<edge_speed_limit_type, int,
						boost::property<edge_lanes_type, int,
							boost::property<edge_divided_type, bool>
						>
					>
				>
			>,
			// Graph properties.
			boost::property<boost::graph_name_t, std::string,
				boost::property<graph_use_right_type, bool,
					boost::property<graph_use_metric_type, bool>
				>
			>
		> map_type;
#endif
		map_type map(5);
		{
			boost::add_edge(0, 1, map);
			boost::add_edge(1, 2, map);
			boost::add_edge(2, 3, map);
			boost::add_edge(3, 4, map);
			boost::add_edge(0, 2, map);
		}

		// Access unbundled properties of vertices, edges, and graph.
		// REF [function] >> graphviz().

		map_type::vertex_descriptor v = *boost::vertices(map).first;
#if defined(__USE_BOOST_PROPERTY_KIND)
		typename boost::property_map<map_type, boost::vertex_name_t>::type vertex_names = boost::get(boost::vertex_name, map);
		typename boost::property_map<map_type, boost::my_vertex_population_type>::type vertex_populations = boost::get(boost::my_vertex_population, map);
#else
		typename boost::property_map<map_type, boost::vertex_name_t>::type vertex_names = boost::get(boost::vertex_name, map);
		typename boost::property_map<map_type, vertex_population_type>::type vertex_populations = boost::get(vertex_population, map);
		//typename boost::property_map<map_type, vertex_population_type>::type vertex_populations = boost::get(vertex_population_type(), map);  // Don't need to define vertex_population.
#endif
		boost::get(vertex_names, v) = "Troy";
		boost::get(vertex_populations, v) = 49170;

		map_type::edge_descriptor e = *boost::edges(map).first;
#if defined(__USE_BOOST_PROPERTY_KIND)
		typename boost::property_map<map_type, boost::my_edge_speed_limit_type>::type edge_speed_limits = boost::get(boost::my_edge_speed_limit, map);
#else
		typename boost::property_map<map_type, edge_speed_limit_type>::type edge_speed_limits = boost::get(edge_speed_limit, map);
		//typename boost::property_map<map_type, edge_speed_limit_type>::type edge_speed_limits = boost::get(edge_speed_limit_type(), map);  // Don't need to define edge_speed_limit.
#endif
		boost::get(edge_speed_limits, e) = 70;

		// Output.
		boost::write_graphviz(
			std::cout, map,
			boost::make_label_writer(vertex_names),  // Vertex properties writer.
			//boost::make_label_writer(vertex_populations),  // Vertex properties writer.
			boost::make_label_writer(edge_speed_limits)  // Edge properties writer.
		);
	}
}

template<class WeightMap, class CapacityMap>
class my_edge_writer
{
public:
	my_edge_writer(WeightMap w, CapacityMap c)
	: wm_(w), cm_(c)
	{}

	template <class Edge>
	void operator()(std::ostream& out, const Edge& e) const
	{
		out << "[weight=\"" << wm_[e] << "\", capacity=\"" << cm_[e] << "\"]";
	}

private:
	WeightMap wm_;
	CapacityMap cm_;
};

template <class WeightMap, class CapacityMap>
inline my_edge_writer<WeightMap, CapacityMap> make_my_edge_writer(WeightMap w, CapacityMap c)
{
	return my_edge_writer<WeightMap, CapacityMap>(w, c);
}

// REF [site] >> http://stackoverflow.com/questions/11369115/how-to-print-a-graph-in-graphviz-with-multiple-properties-displayed
// REF [site] >> http://stackoverflow.com/questions/9181183/how-to-print-a-boost-graph-in-graphviz-with-one-of-the-properties-displayed
void bundled_properties_2()
{
	struct VertexProperty
	{
		std::string name;
	};

	struct EdgeProperty
	{
		int capacity;
		int weight;
	};

	typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty> graph_type;
	graph_type g;
	{
		graph_type::vertex_descriptor v0 = boost::add_vertex(g);
		graph_type::vertex_descriptor v1 = boost::add_vertex(g);
		graph_type::vertex_descriptor v2 = boost::add_vertex(g);
		graph_type::vertex_descriptor v3 = boost::add_vertex(g);
		graph_type::vertex_descriptor v4 = boost::add_vertex(g);

		auto vertex_names = boost::get(&VertexProperty::name, g);
		boost::get(vertex_names, v0) = "v0";
		boost::get(vertex_names, v1) = "v1";
		boost::get(vertex_names, v2) = "v2";
		boost::get(vertex_names, v3) = "v3";
		boost::get(vertex_names, v4) = "v4";

		EdgeProperty prop;
		prop.weight = 10;
		prop.capacity = -10;
		boost::add_edge(v0, v1, prop, g);
		prop.weight = 20;
		prop.capacity = -20;
		boost::add_edge(v1, v2, prop, g);
		prop.weight = 30;
		prop.capacity = -30;
		boost::add_edge(v2, v3, prop, g);
		prop.weight = 40;
		prop.capacity = -40;
		boost::add_edge(v1, v3, prop, g);
		prop.weight = 50;
		prop.capacity = -50;
		boost::add_edge(v1, v4, prop, g);
		prop.weight = 60;
		prop.capacity = -60;
		boost::add_edge(v0, v3, prop, g);
		prop.weight = 70;
		prop.capacity = -70;
		boost::add_edge(v0, v4, prop, g);
	}

#if 0
	{
		const auto &names = boost::get(&VertexProperty::name, g);

		std::cout << "Vertex set: ";
		boost::print_vertices(g, names);
		std::cout << "Edge set: ";
		boost::print_edges(g, names);
		std::cout << "Outgoing edges:" << std::endl;
		boost::print_graph(g, names);
	}
#endif

	//std::ofstream dot("./data/boost/graph.dot");
#if 0
	boost::write_graphviz(
		std::cout, g,
		boost::make_label_writer(boost::get(&VertexProperty::name, g)),  // Vertex properties writer.
		boost::make_label_writer(boost::get(&EdgeProperty::weight, g))  // Edge properties writer.
	);
#elif 0
	boost::write_graphviz(
		std::cout, g,
		boost::make_label_writer(boost::get(&VertexProperty::name, g)),  // Vertex properties writer.
		boost::make_label_writer(boost::get(&EdgeProperty::capacity, g))  // Edge properties writer.
	);
#else
	boost::write_graphviz(
		std::cout, g,
		boost::make_label_writer(boost::get(&VertexProperty::name, g)),  // Vertex properties writer.
		make_my_edge_writer(boost::get(&EdgeProperty::weight, g), boost::get(&EdgeProperty::capacity, g))  // Edge properties writer.
	);
#endif
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/graph/doc/adjacency_matrix.html
void graph_based_on_adjacency_matrix()
{
	// Directed graph based on boost::adjacency_matrix.
	{
		enum { A, B, C, D, E, F, N };
		const char* name = "ABCDEF";

		typedef boost::adjacency_matrix<boost::directedS> Graph;
		Graph g(N);
		boost::add_edge(B, C, g);
		boost::add_edge(B, F, g);
		boost::add_edge(C, A, g);
		boost::add_edge(C, C, g);
		boost::add_edge(D, E, g);
		boost::add_edge(E, D, g);
		boost::add_edge(F, A, g);

		std::cout << "Vertex set: ";
		boost::print_vertices(g, name);
		std::cout << std::endl;

		std::cout << "Edge set: ";
		boost::print_edges(g, name);
		std::cout << std::endl;

		std::cout << "Outgoing edges: " << std::endl;
		boost::print_graph(g, name);
		std::cout << std::endl;
	}

	// Undirected graph based on boost::adjacency_matrix.
	{
		enum { A, B, C, D, E, F, N };
		const char* name = "ABCDEF";

		typedef boost::adjacency_matrix<boost::undirectedS> UGraph;
		UGraph ug(N);
		boost::add_edge(B, C, ug);
		boost::add_edge(B, F, ug);
		boost::add_edge(C, A, ug);
		boost::add_edge(D, E, ug);
		boost::add_edge(F, A, ug);

		std::cout << "Vertex set: ";
		boost::print_vertices(ug, name);
		std::cout << std::endl;

		std::cout << "Edge set: ";
		boost::print_edges(ug, name);
		std::cout << std::endl;

		std::cout << "Incident edges: " << std::endl;
		boost::print_graph(ug, name);
		std::cout << std::endl;
	}
}

void default_undirected_and_directed_graph()
{
	// boost::undirected_graph<>.
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
		std::cout << "Degree of u: " << boost::degree(u, g) << std::endl;

		//
		boost::undirected_graph<>::vertex_iterator vertexIt, vertexEnd;  // Iterate over all the vertices of the graph.
		boost::undirected_graph<>::adjacency_iterator neighborIt, neighborEnd;  // Iterate over the corresponding adjacent vertices.
		boost::undirected_graph<>::out_edge_iterator edgeIt, edgeEnd;  // Iterate over all the incident edges of a vertex.
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			// Access adjacent vertices of each vertex.
			std::cout << *vertexIt << " is connected with ";
			boost::tie(neighborIt, neighborEnd) = boost::adjacent_vertices(*vertexIt, g);
			for (; neighborIt != neighborEnd; ++neighborIt)
				std::cout << *neighborIt << " ";
			std::cout << std::endl;

			// Access incident edges of each vertex.
			std::cout << *vertexIt << " is incident to ";
			boost::tie(edgeIt, edgeEnd) = boost::incident_edges(*vertexIt, g);
			for (; edgeIt != edgeEnd; ++edgeIt)
				std::cout << *edgeIt << " ";
			std::cout << std::endl;
		}
	}

	// boost::directed_graph<>.
	{
		boost::directed_graph<> g;

		// TODO [add] >>
	}
}

// Define a simple function to print vertices.
template<typename Graph>
void print_vertex(typename boost::graph_traits<Graph>::vertex_descriptor vertex_to_print)
{
	std::cout << "(" << vertex_to_print[0] << ", " << vertex_to_print[1] << ", " << vertex_to_print[2] << ")" << std::endl;
}

// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/graph/example/grid_graph_example.cpp
void grid_graph()
{
	const int DIMENSIONS = 3;

	typedef boost::grid_graph<DIMENSIONS> graph_type;
	typedef boost::graph_traits<graph_type> traits_type;

	// Define a 3x5x7 grid_graph where the second dimension doesn't wrap.
	boost::array<std::size_t, 3> lengths = { { 3, 5, 7 } };
	boost::array<bool, 3> wrapped = { { true, false, true } };
	graph_type graph(lengths, wrapped);

	// Do a round-trip test of the vertex index functions.
	for (traits_type::vertices_size_type v_index = 0; v_index < num_vertices(graph); ++v_index)
	{
		// The two indicies should always be equal.
		std::cout << "Index of vertex " << v_index << " is " << get(boost::vertex_index, graph, vertex(v_index, graph)) << std::endl;
	}

	// Do a round-trip test of the edge index functions.
	for (traits_type::edges_size_type e_index = 0; e_index < num_edges(graph); ++e_index)
	{
		// The two indicies should always be equal.
		std::cout << "Index of edge " << e_index << " is " << get(boost::edge_index, graph, edge_at(e_index, graph)) << std::endl;
	}

	// Print number of dimensions.
	std::cout << graph.dimensions() << std::endl;  // Prints "3".

	// Print dimension lengths (same order as in the lengths array).
	std::cout << graph.length(0) << "x" << graph.length(1) << "x" << graph.length(2) << std::endl;  // Prints "3x5x7".

	// Print dimension wrapping (W = wrapped, U = unwrapped).
	std::cout << (graph.wrapped(0) ? "W" : "U") << ", " << (graph.wrapped(1) ? "W" : "U") << ", " << (graph.wrapped(2) ? "W" : "U") << std::endl;  // Prints "W, U, W".

	// Start with the first vertex in the graph.
	traits_type::vertex_descriptor first_vertex = vertex(0, graph);
	print_vertex<graph_type>(first_vertex);  // Prints "(0, 0, 0)".

	// Print the next vertex in dimension 0.
	print_vertex<graph_type>(graph.next(first_vertex, 0));  // Prints "(1, 0, 0)".

	// Print the next vertex in dimension 1.
	print_vertex<graph_type>(graph.next(first_vertex, 1));  // Prints "(0, 1, 0)".

	// Print the 5th next vertex in dimension 2.
	print_vertex<graph_type>(graph.next(first_vertex, 2, 5));  // Prints "(0, 0, 4)".

	// Print the previous vertex in dimension 0 (wraps).
	print_vertex<graph_type>(graph.previous(first_vertex, 0));  // Prints "(2, 0, 0)".

	// Print the previous vertex in dimension 1 (doesn't wrap, so it's the same).
	print_vertex<graph_type>(graph.previous(first_vertex, 1));  // Prints "(0, 0, 0)".

	// Print the 20th previous vertex in dimension 2 (wraps around twice).
	print_vertex<graph_type>(graph.previous(first_vertex, 2, 20));  // Prints "(0, 0, ?)".
}

void graphviz()
{
	// Read graphviz.
	// REF [file] >> ${BOOST_HOME}/libs/graph/example/read_graphviz.cpp
	{
		// REF [function] >> bundled_properties_1().

		// Vertex properties.
		typedef boost::property<boost::vertex_name_t, std::string, boost::property<boost::vertex_color_t, float> > vertex_p;
		// Edge properties.
		typedef boost::property<boost::edge_weight_t, double> edge_p;
		// Graph properties.
		typedef boost::property<boost::graph_name_t, std::string> graph_p;
		// Adjacency_list-based type.
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_p, edge_p, graph_p> graph_type;

		// Construct an empty graph and prepare the dynamic_property_maps.
		graph_type graph(0);
		boost::dynamic_properties dp;

		boost::property_map<graph_type, boost::vertex_name_t>::type name(boost::get(boost::vertex_name, graph));
		dp.property("node_id", name);

		boost::property_map<graph_type, boost::vertex_color_t>::type mass(boost::get(boost::vertex_color, graph));
		dp.property("mass", mass);

		boost::property_map<graph_type, boost::edge_weight_t>::type weight(boost::get(boost::edge_weight, graph));
		dp.property("weight", weight);

		// Use ref_property_map to turn a graph property into a property map.
		boost::ref_property_map<graph_type *, std::string> gname(boost::get_property(graph, boost::graph_name));
		dp.property("name", gname);

		// Sample graph as an std::istream.
		std::istringstream gvgraph_stream("digraph { graph [name=\"graphname\"]  a  c e [mass = 6.66] }");

		const bool status = boost::read_graphviz(gvgraph_stream, graph, dp, "node_id");

		boost::write_graphviz(std::cout, graph);
		boost::write_graphviz_dp(std::cout, graph, dp, "node_id");  // == boost::write_graphviz_dp(std::cout, graph, dp);
		boost::write_graphviz_dp(std::cout, graph, dp, "mass");
		//boost::write_graphviz_dp(std::cout, graph, dp, "weight");  // exception: dynamic property get cannot retrieve value for property: weight.
		//boost::write_graphviz_dp(std::cout, graph, dp, "name");  // exception: dynamic property get cannot retrieve value for property: name.
	}

	// Write graphviz.
	// REF [file] >> ${BOOST_HOME}/libs/graph/example/write_graphviz.cpp
	{
		enum files_e {
			dax_h, yow_h, boz_h, zow_h, foo_cpp,
			foo_o, bar_cpp, bar_o, libfoobar_a,
			zig_cpp, zig_o, zag_cpp, zag_o,
			libzigzag_a, killerapp, N
		};
		const char* name[] = {
			"dax.h", "yow.h", "boz.h", "zow.h", "foo.cpp",
			"foo.o", "bar.cpp", "bar.o", "libfoobar.a",
			"zig.cpp", "zig.o", "zag.cpp", "zag.o",
			"libzigzag.a", "killerapp"
		};

		typedef std::pair<int, int> Edge;
		Edge used_by[] = {
			Edge(dax_h, foo_cpp), Edge(dax_h, bar_cpp), Edge(dax_h, yow_h),
			Edge(yow_h, bar_cpp), Edge(yow_h, zag_cpp),
			Edge(boz_h, bar_cpp), Edge(boz_h, zig_cpp), Edge(boz_h, zag_cpp),
			Edge(zow_h, foo_cpp),
			Edge(foo_cpp, foo_o),
			Edge(foo_o, libfoobar_a),
			Edge(bar_cpp, bar_o),
			Edge(bar_o, libfoobar_a),
			Edge(libfoobar_a, libzigzag_a),
			Edge(zig_cpp, zig_o),
			Edge(zig_o, libzigzag_a),
			Edge(zag_cpp, zag_o),
			Edge(zag_o, libzigzag_a),
			Edge(libzigzag_a, killerapp)
		};
		const int nedges = sizeof(used_by) / sizeof(Edge);
		int weights[nedges];
		std::fill(weights, weights + nedges, 1);

		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::property<boost::vertex_color_t, boost::default_color_type>, boost::property<boost::edge_weight_t, int> > graph_type;
		graph_type g(used_by, used_by + nedges, weights, N);

		boost::write_graphviz(std::cout, g, boost::make_label_writer(name));
		//boost::write_graphviz(std::cout, g, boost::make_label_writer(name), boost::make_label_writer(name));
	}
}

}  // namespace boost_graph
