#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/max_cardinality_matching.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${BOOST_HOME}/libs/graph/example/max_flow.cpp
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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/push-relabel-eg.cpp
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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/boykov_kolmogorov-eg.cpp
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

// REF [file] >> ${BOOST_HOME}/libs/graph/example/matching_example.cpp
void edmonds_maximum_cardinality_matching_example()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

	// Create the following graph: (it'll look better when output to the terminal in a fixed width font...)
	const int num_vertices = 18;

	std::vector<std::string> ascii_graph;
	ascii_graph.push_back("           0       1---2       3       ");
	ascii_graph.push_back("            \\     /     \\     /        ");
	ascii_graph.push_back("             4---5       6---7         ");
	ascii_graph.push_back("             |   |       |   |         ");
	ascii_graph.push_back("             8---9      10---11        ");
	ascii_graph.push_back("            /     \\     /     \\        ");
	ascii_graph.push_back("     12   13      14---15      16   17 ");

	// It has a perfect matching of size 8. There are two isolated vertices that we'll use later...
	graph_type g(num_vertices);

	// Our vertices are stored in a vector, so we can refer to vertices by integers in the range 0..15
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

	// Find the maximum cardinality matching.
	// We'll use a checked version of the algorithm, which takes a little longer than the unchecked version,
	// but has the advantage that it will return "false" if the matching returned is not actually a maximum cardinality matching in the graph.
#if 0
	boost::edmonds_maximum_cardinality_matching(g, &mate[0]);
#else
	const bool success1 = checked_edmonds_maximum_cardinality_matching(g, &mate[0]);
	assert(success1);
#endif

	std::cout << "In the following graph:" << std::endl << std::endl;
	for (std::vector<std::string>::iterator itr = ascii_graph.begin(); itr != ascii_graph.end(); ++itr)
		std::cout << *itr << std::endl;

	std::cout << std::endl << "Found a matching of size " << boost::matching_size(g, &mate[0]) << std::endl;
	std::cout << "The matching is:" << std::endl;
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

	std::cout << "In the following graph:" << std::endl << std::endl;
	for (std::vector<std::string>::iterator itr = ascii_graph.begin(); itr != ascii_graph.end(); ++itr)
		std::cout << *itr << std::endl;

	std::cout << std::endl << "Found a matching of size " << boost::matching_size(g, &mate[0]) << std::endl;
	std::cout << "The matching is:" << std::endl;
	for (boost::tie(vi, vi_end) = boost::vertices(g); vi != vi_end; ++vi)
		if (mate[*vi] != boost::graph_traits<graph_type>::null_vertex() && *vi < mate[*vi])
			std::cout << "{" << *vi << ", " << mate[*vi] << "}" << std::endl;
}

struct edge_t
{
	unsigned long first;
	unsigned long second;
};

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

void maximum_flow_and_matching()
{
	const std::string max_flow_dat_file("./data/boost/max_flow.dat");
#if defined(__GNUC__)
	std::ifstream stream(max_flow_dat_file.c_str());
#else
	std::ifstream stream(max_flow_dat_file);
#endif

	std::cout << "Max-flow algorithm -------------------------------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) local::max_flow_example(stream);

	std::cout << "\nPush–relabel maximum flow algorithm --------------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) local::push_relabel_example(stream);

	std::cout << "\nBoykov-Kolmogorov (BK) max-flow algorithm --------------------" << std::endl;
	stream.clear();
	stream.seekg(0, std::ios::beg);
	if (stream.is_open()) local::boykov_kolmogorov_example(stream);

	std::cout << "\nEdmonds maximum cardinality matching -------------------------" << std::endl;
	local::edmonds_maximum_cardinality_matching_example();
}

void minimum_cut()
{
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int> > undirected_graph_type;
	//typedef boost::graph_traits<undirected_graph_type>::vertex_descriptor vertex_descriptor_type;
	typedef boost::property_map<undirected_graph_type, boost::edge_weight_t>::type weight_map_type;
	typedef boost::property_traits<weight_map_type>::value_type weight_type;

	// Define the 16 edges of the graph. {3, 4} means an undirected edge between vertices 3 and 4.
	const local::edge_t edges[] = {
		{3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7}, {0, 5},
		{0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}
	};

	// For each of the 16 edges, define the associated edge weight.
	// ws[i] is the weight for the edge that is described by edges[i].
	const weight_type ws[] = { 0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4 };

	// Construct the graph object.
	// 8 is the number of vertices, which are numbered from 0 // through 7, and 16 is the number of edges.
	undirected_graph_type g(edges, edges + 16, ws, 8, 16);

	// Define a property map, 'parities', that will store a boolean value for each vertex.
	// Vertices that have the same parity after 'stoer_wagner_min_cut' runs are on the same side of the min-cut.
	BOOST_AUTO(parities, boost::make_one_bit_color_map(boost::num_vertices(g), boost::get(boost::vertex_index, g)));

	// Run the Stoer-Wagner algorithm to obtain the min-cut weight. `parities` is also filled in.
	const int w = boost::stoer_wagner_min_cut(g, boost::get(boost::edge_weight, g), boost::parity_map(parities));

	std::cout << "The min-cut weight of G is " << w << ".\n" << std::endl;
	assert(w == 7);

	std::cout << "One set of vertices consists of:" << std::endl;
	std::size_t i;
	for (i = 0; i < boost::num_vertices(g); ++i)
	{
		if (boost::get(parities, i))
			std::cout << i << std::endl;
	}
	std::cout << std::endl;

	std::cout << "The other set of vertices consists of:" << std::endl;
	for (i = 0; i < boost::num_vertices(g); ++i)
	{
		if (!boost::get(parities, i))
			std::cout << i << std::endl;
	}
	std::cout << std::endl;
}

}  // namespace boost_graph
