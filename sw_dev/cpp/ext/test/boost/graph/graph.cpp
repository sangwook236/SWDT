#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/strong_components.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/incremental_components.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/utility.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/foreach.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <utility>
#include <stdexcept>


namespace {
namespace local {

void other_core_algorithms()
{
	std::cout << "\nTopological sort ---------------------------------------------" << std::endl;
	{
		//boost::topological_sort();
		throw std::runtime_error("Not yet implemented");
	}

	std::cout << "\nTransitive closure -------------------------------------------" << std::endl;
	{
		//boost::transitive_closure();
		throw std::runtime_error("Not yet implemented");
	}
}

// REF [file] >> ${BOOST_HOME}/libs/graph/example/connected_components.cpp
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
		std::cout << "Total number of components: " << num << std::endl;
		for (std::vector<int>::size_type i = 0; i != components.size(); ++i)
		  std::cout << "Vertex " << i <<" is in component " << components[i] << std::endl;
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
		std::cout << "Total number of components: " << num << std::endl;
		for (std::vector<int>::iterator i = c.begin(); i != c.end(); ++i)
			std::cout << "Vertex " << i - c.begin() << " is in component " << *i << std::endl;
	}
}

// REF [file] >> ${BOOST_HOME}/libs/graph/example/connected_components.cpp
void connected_components_test()
{
	std::cout << "------------------------------------------------------------" << std::endl;
	{
		const int N = 6;

#if 0
		using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

		Graph G(N);
		boost::add_edge(0, 1, G);
		boost::add_edge(1, 4, G);
		boost::add_edge(4, 0, G);
		boost::add_edge(2, 5, G);
#else
		using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, float>>;
		//using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, int, boost::property<boost::edge_weight_t, float>>;
		//using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_color_t, int>, boost::property<boost::edge_weight_t, float>>;

		Graph G(N);
		boost::add_edge(0, 1, 1.2f, G);
		boost::add_edge(1, 4, 3.0f, G);
		boost::add_edge(4, 0, 2.5f, G);
		boost::add_edge(2, 5, 0.7f, G);
#endif

		{
			std::vector<int> components(boost::num_vertices(G));
			//const auto &num_components = boost::connected_components(G, &components[0]);
			const auto &num_components = boost::connected_components(G, boost::make_iterator_property_map(components.begin(), boost::get(boost::vertex_index, G), components[0]));

			std::cout << "Total number of components = " << num_components << std::endl;
			for (auto it = components.begin(); it != components.end(); ++it)
				std::cout << "Vertex " << (it - components.begin()) << " is in component " << *it << std::endl;
			std::cout << std::endl;
		}

		{
			std::map<Graph::vertex_descriptor, int> components;
			const auto &num_components = boost::connected_components(G, boost::make_assoc_property_map(components));

			std::cout << "Total number of components = " << num_components << std::endl;
			for (auto it = components.begin(); it != components.end(); ++it)
				std::cout << "Vertex " << it->first << " is in component " << it->second << std::endl;
			std::cout << std::endl;
		}
	}

	std::cout << "------------------------------------------------------------" << std::endl;
	{
		const int N = 6;

#if 1
		using Graph = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS>;

		Graph G;
		for (int i = 0; i < N; ++i)
			boost::add_vertex(G);

		boost::add_edge(boost::vertex(0, G), boost::vertex(1, G), G);
		boost::add_edge(boost::vertex(1, G), boost::vertex(4, G), G);
		boost::add_edge(boost::vertex(4, G), boost::vertex(0, G), G);
		boost::add_edge(boost::vertex(2, G), boost::vertex(5, G), G);
#else
		using Graph = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, float>>;
		//using Graph = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, int, boost::property<boost::edge_weight_t, float>>;
		//using Graph = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, boost::property<boost::vertex_color_t, int>, boost::property<boost::edge_weight_t, float>>;

		Graph G;
		for (int i = 0; i < N; ++i)
			boost::add_vertex(G);

		boost::add_edge(boost::vertex(0, G), boost::vertex(1, G), 1.2f, G);
		boost::add_edge(boost::vertex(1, G), boost::vertex(4, G), 3.0f, G);
		boost::add_edge(boost::vertex(4, G), boost::vertex(0, G), 2.5f, G);
		boost::add_edge(boost::vertex(2, G), boost::vertex(5, G), 0.7f, G);
#endif

#if 0
		{
			std::vector<int> components(boost::num_vertices(G));
			//const auto &num_components = boost::connected_components(G, &components[0]);  // Compile-time error: 'put': no matching overloaded function found
			//const auto &num_components = boost::connected_components(G, boost::make_iterator_property_map(components.begin(), boost::get(boost::vertex_index, G), components[0]));
			auto &prop_map = boost::make_iterator_property_map(components.begin(), boost::get(boost::vertex_index, G), components[0]);  // Compile-time error: you cannot create a reference to 'void'
			const auto &num_components = boost::connected_components(G, prop_map);

			std::cout << "Total number of components = " << num_components << std::endl;
			for (auto it = components.begin(); it != components.end(); ++it)
				std::cout << "Vertex " << (it - components.begin()) << " is in component " << *it << std::endl;
			std::cout << std::endl;
		}
#endif

#if 0
		{
			std::map<Graph::vertex_descriptor, int> components;
			const auto &num_components = boost::connected_components(G, boost::make_assoc_property_map(components));  // Compile-time error: you cannot create a reference to 'void'

			std::cout << "Total number of components = " << num_components << std::endl;
			for (auto it = components.begin(); it != components.end(); ++it)
				std::cout << "Vertex " << it->first << " is in component " << it->second << std::endl;
			std::cout << std::endl;
		}
#endif
	}
}

// REF [file] >> ${BOOST_HOME}/libs/graph/example/strong_components.cpp
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

		std::cout << "Total number of components: " << num << std::endl;
		for (std::vector<int>::iterator i = components.begin(); i != components.end(); ++i)
			std::cout << "Vertex " << i - components.begin() << " is in component " << *i << std::endl;
	}
/*
	{
		// Vertex properties.
		typedef boost::property<boost::vertex_name_t, std::string> vertex_p;
		// Edge properties.
		typedef boost::property<boost::edge_weight_t, double> edge_p;
		// Graph properties.
		typedef boost::property<boost::graph_name_t, std::string, boost::property<boost::graph_graph_attribute_t, float> > graph_p;
		// Adjacency_list-based type.
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, vertex_p, edge_p, graph_p> graph_type;
		//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> graph_type;

		graph_type g(0);
		boost::dynamic_properties dp;

		{
			// Compile-time error

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

		std::cout << "Total number of components: " << num << std::endl;
		for (std::vector<int>::size_type i = 0; i != components.size(); ++i)
			std::cout << "Vertex " << name[i] <<" is in component " << components[i] << std::endl;
	}
*/
}

struct edge_component_t
{
	enum { num = 555 };
	typedef boost::edge_property_tag kind;
} edge_component;

// REF [file] >> ${BOOST_HOME}/libs/graph/example/biconnected_components.cpp
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
	std::cout << "Found " << num_comps << " biconnected components." << std::endl;

	//
	std::vector<vertex_descriptor_type> art_points;
	boost::articulation_points(g, std::back_inserter(art_points));
	std::cout << "Found " << art_points.size() << " articulation points." << std::endl;

	//
	std::cout << "graph A {\n" << "  node[shape=\"circle\"]" << std::endl;
	for (std::size_t i = 0; i < art_points.size(); ++i)
		std::cout << (char)(art_points[i] + 'A') << " [ style=\"filled\", fillcolor=\"red\" ];" << std::endl;

	boost::graph_traits<graph_type>::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
		std::cout << (char)(boost::source(*ei, g) + 'A') << " -- " << (char)(boost::target(*ei, g) + 'A') << "[label=\"" << components[*ei] << "\"]" << std::endl;
	std::cout << '}' << std::endl;
}

// REF [file] >> ${BOOST_HOME}/libs/graph/example/incremental-components-eg.cpp
// REF [file] >> ${BOOST_HOME}/libs/graph/example/incremental_components.cpp
void incremental_connected_components_algorithm()
{
/*
	{
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;
		typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_descriptor_type;
		typedef boost::graph_traits<graph_type>::edge_descriptor edge_descriptor_type;
		typedef boost::graph_traits<graph_type>::vertices_size_type vertex_index_type;

		// Create a graph.
		const int VERTEX_COUNT = 6;
		graph_type graph(VERTEX_COUNT);

		boost::add_edge(0, 1, graph);
		boost::add_edge(1, 4, graph);

		// Create the disjoint-sets object, which requires rank and parent vertex properties.
		std::vector<vertex_descriptor_type> rank(boost::num_vertices(graph));
		std::vector<vertex_descriptor_type> parent(boost::num_vertices(graph));

		typedef vertex_index_type * rank_type;
		typedef vertex_descriptor_type * parent_type;
		boost::disjoint_sets<rank_type, parent_type> ds(&rank[0], &parent[0]);

		// Determine the connected components, storing the results in the disjoint-sets object.
		boost::initialize_incremental_components(graph, ds);
		boost::incremental_components(graph, ds);

		// Add a couple more edges and update the disjoint-sets.
		boost::add_edge(4, 0, graph);
		boost::add_edge(2, 5, graph);

		ds.union_set(4, 0);
		ds.union_set(2, 5);

		BOOST_FOREACH(vertex_descriptor_type current_vertex, boost::vertices(graph))
			std::cout << "Representative[" << current_vertex << "] = " << ds.find_set(current_vertex) << std::endl;
		std::cout << std::endl;

		// Generate component index.
		// NOTE: We would need to pass in a vertex index map into the component_index constructor
		// if our graph type used listS instead of vecS (identity_property_map is used by default).
		typedef boost::component_index<vertex_index_type> components_type;
		components_type components(parent.begin(), parent.end());

		// Iterate through the component indices.
		BOOST_FOREACH(vertex_index_type component_index, components)
		{
			std::cout << "Component " << component_index << " contains: ";

			// Iterate through the child vertex indices for [component_index].
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

		std::cout << "An undirected graph:" << std::endl;
		boost::print_graph(graph, boost::get(boost::vertex_index, graph));
		std::cout << std::endl;

		BOOST_FOREACH(vertex_descriptor_type current_vertex, boost::vertices(graph))
			std::cout << "Representative[" << current_vertex << "] = " << ds.find_set(current_vertex) << std::endl;
		std::cout << std::endl;

		typedef boost::component_index<vertex_index_type> components_type;

		// NOTE: Because we're using vecS for the graph type, we're effectively using identity_property_map for a vertex index map.
		// If we were to use listS instead, the index map would need to be explicitly passed to the component_index constructor.
		components_type components(parent.begin(), parent.end());

		// Iterate through the component indices.
		BOOST_FOREACH(vertex_index_type current_index, components)
		{
			std::cout << "Component " << current_index << " contains: ";

			// Iterate through the child vertex indices for [current_index].
			BOOST_FOREACH(vertex_index_type child_index, components[current_index])
				std::cout << child_index << " ";
			std::cout << std::endl;
		}
	}
*/
}

void connected_components()
{
	std::cout << "Connected components algorithm --------------------------------" << std::endl;
	connected_components_algorithm();
	connected_components_test();

	std::cout << "\nStrong components algorithm -----------------------------------" << std::endl;
	strong_components_algorithm();

	std::cout << "\nBiconnected components algorithm ------------------------------" << std::endl;
	biconnected_components_algorithm();

	std::cout << "\nIncremental connected components algorithm --------------------" << std::endl;
	incremental_connected_components_algorithm();
}

void sparse_matrix_ordering()
{
	throw std::runtime_error("Not yet implemented");
}

void layout_algorithms()
{
	throw std::runtime_error("Not yet implemented");
}

void clustering()
{
	throw std::runtime_error("Not yet implemented");
}

void planar_graph_algorithms()
{
	throw std::runtime_error("Not yet implemented");
}

void graph_metrics()
{
	throw std::runtime_error("Not yet implemented");
}

void graph_structure_comparisons()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

void boost_quick_tour();
void basic_operation_1();
void basic_operation_2();
void basic_operation_3();

void bundled_properties_1();
void bundled_properties_2();
void graph_based_on_adjacency_matrix();
void default_undirected_and_directed_graph();
void grid_graph();
void graphviz();

void traversal();
void shortest_paths();
void minimum_spanning_tree();
void random_spanning_tree();
void common_spanning_tree();
void maximum_flow_and_matching();
void minimum_cut();
void metric_tsp_approximation();

}  // namespace boost_graph

void graph()
{
	// Basic.
	{
		std::cout << "Quick tour ---------------------------------------------------" << std::endl;
		// Access and iterate vertices, edges, and their properties.
		//boost_graph::boost_quick_tour();

		std::cout << "\nBasic operation ----------------------------------------------" << std::endl;
		// Undirected, directed, and bidirectional graphs based on boost::adjacency_list.
		// boost::undirected_graph<> and boost::directed_graph<>.
		// Access adjacent vertices, incoming and outgoing edges of a vertex.
		// Access the degree, in- & out-degree of a vertex and the source and target of an edge.
		boost_graph::basic_operation_1();
		//boost_graph::basic_operation_2();
		//boost_graph::basic_operation_3();

		std::cout << "\nBundled properties -------------------------------------------" << std::endl;
		// Bundled(user-defined) properties of vertex, edge, or graph.
		//boost_graph::bundled_properties_1();
		//boost_graph::bundled_properties_2();

		std::cout << "\nGraph based on adjacency matrix ------------------------------" << std::endl;
		// Undirected and directed graphs based on boost::adjacency_matrix.
		// Use boost::print_vertices, boost::print_edges, and boost::print_graph.
		//boost_graph::graph_based_on_adjacency_matrix();

		std::cout << "\nboost::undirected_graph<> and boost::directed_graph<> --------" << std::endl;
		// boost::undirected_graph<> and boost::directed_graph<>.
		//boost_graph::default_undirected_and_directed_graph();

		std::cout << "\nGrid graph ---------------------------------------------------" << std::endl;
		// Grid graph.
		//boost_graph::grid_graph();

		std::cout << "\nGraphviz -----------------------------------------------------" << std::endl;
		// Use graphviz.
		boost_graph::graphviz();
	}

	// Algorithm.
	{
		std::cout << "\nTraversal algorithms -----------------------------------------" << std::endl;
		//boost_graph::traversal();

		std::cout << "\nShortest paths / cost minimization algorithms ----------------" << std::endl;
		//boost_graph::shortest_paths();

		std::cout << "\nOther core algorithms ----------------------------------------" << std::endl;
		//local::other_core_algorithms();  // Not yet implemented.

		std::cout << "\nMinimum spanning tree algorithms -----------------------------" << std::endl;
		//boost_graph::minimum_spanning_tree();

		std::cout << "\nRandom spanning tree algorithms ------------------------------" << std::endl;
		//boost_graph::random_spanning_tree();  // Not yet implemented.

		std::cout << "\nAlgorithm for common spanning trees of two graphs ------------" << std::endl;
		//boost_graph::common_spanning_tree();  // Not yet implemented.

		std::cout << "\nConnected components algorithms ------------------------------" << std::endl;
		//local::connected_components();

		std::cout << "\nMaximum flow and matching algorithms -------------------------" << std::endl;
		//boost_graph::maximum_flow_and_matching();

		std::cout << "\nMinimum cut algorithms ---------------------------------------" << std::endl;
		//boost_graph::minimum_cut();

		std::cout << "\nSparse matrix ordering algorithms ----------------------------" << std::endl;
		//local::sparse_matrix_ordering();  // Not yet implemented.

		std::cout << "\nLayout algorithms --------------------------------------------" << std::endl;
		//local::layout_algorithms();  // Not yet implemented.

		std::cout << "\nClustering algorithms ----------------------------------------" << std::endl;
		//local::clustering();  // Not yet implemented.

		std::cout << "\nPlanar graph algorithms --------------------------------------" << std::endl;
		//local::planar_graph_algorithms();  // Not yet implemented.

		std::cout << "\nGraph metrics ------------------------------------------------" << std::endl;
		//local::graph_metrics();  // Not yet implemented.

		std::cout << "\nGraph structure comparisons ----------------------------------" << std::endl;
		//local::graph_structure_comparisons();  // Not yet implemented.
	}

	// Application.
	{
		// Traveling salesperson problem (TSP).
		//boost_graph::metric_tsp_approximation();
	}
}
