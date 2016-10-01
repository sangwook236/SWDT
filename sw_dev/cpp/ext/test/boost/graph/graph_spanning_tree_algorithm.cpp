#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${BOOST_HOME}/libs/graph/example/kruskal-example.cpp
void kruskal_minimum_spanning_tree_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;
	typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
	//typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/kruskal-telephone.cpp
void kruskal_minimum_spanning_tree_telephone_example()
{
/*
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property,boost::property<boost::edge_weight_t, int> > graph_type;

#if 0
	// REF [file] >> GraphvizGraph was defined in boost/graph/graphviz.hpp, but cannot be used at present

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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/prim-example.cpp
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
	//boost::property_map<graph_type, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, g);
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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/prim-telephone.cpp
void prim_minimum_spanning_tree_telephone_example()
{
/*
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > graph_type;

#if 0
	// REF [file] >> GraphvizGraph was defined in boost/graph/graphviz.hpp, but cannot be used at present

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

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

void minimum_spanning_tree()
{
	std::cout << "Kruskal's minimum spanning tree algorithm ---------------------" << std::endl;
	local::kruskal_minimum_spanning_tree_example();
	//local::kruskal_minimum_spanning_tree_telephone_example();  // Compile-time error.

	std::cout << "\nPrim's minimum spanning tree algorithm ------------------------" << std::endl;
	local::prim_minimum_spanning_tree_example();
	//local::prim_minimum_spanning_tree_telephone_example();  // Compile-time error.
}

void random_spanning_tree()
{
	throw std::runtime_error("Not yet implemented");
}

void common_spanning_tree()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace boost_graph
