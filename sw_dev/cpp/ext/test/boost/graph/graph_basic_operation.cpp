#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
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


#define __USE_BOOST_PROPERTY_KIND 1

#if defined(__USE_BOOST_PROPERTY_KIND)
namespace boost {

enum my_population_type { my_population };
enum my_zipcodes_type { my_zipcodes };
template <>
struct property_kind<my_population_type>
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
		typename boost::property_map<Graph, boost::vertex_index_t>::type vertex_id = boost::get(boost::vertex_index, g_);

		std::cout << "vertex: " << name_[boost::get(vertex_id, v)] << std::endl;

		// Write out the outgoing edges.
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

		// Write out the incoming edges.
		std::cout << "\tin-edges: ";
		typename boost::graph_traits<Graph>::in_edge_iterator in_i, in_end;
		for (boost::tie(in_i, in_end) = boost::in_edges(v, g_); in_i != in_end; ++in_i)
		{
			e = *in_i;
			vertex_descriptor_type src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_id, src)] << "," << name_[boost::get(vertex_id, targ)] << ") ";
		}
		std::cout << std::endl;

		// Write out all adjacent vertices.
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

		graph_type::vertex_iterator vertexIt, vertexEnd;  // Iterate over all the vertices of the graph.
		graph_type::adjacency_iterator neighbourIt, neighbourEnd;  // Iterate over the corresponding adjacent vertices.

		// Access adjacent vertices of each vertex.
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
		// When using the directedS tag in BGL, you are allowed to use only the out_edges helper function and associated iterators.
		// Using in_edges requires changing the graph type to bidirectionalS, although this is still more or less a directed graph.
		//typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS> graph_type;  // Compile-time error.
		typedef boost::adjacency_list<boost::listS, boost::vecS, boost::bidirectionalS> graph_type;

		graph_type g;
		boost::add_edge(0, 1, g);
		boost::add_edge(0, 3, g);
		boost::add_edge(1, 2, g);
		boost::add_edge(2, 3, g);

		graph_type::vertex_iterator vertexIt, vertexEnd;
		graph_type::in_edge_iterator inedgeIt, inedgeEnd;

		// Access incoming edges of each vertex.
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

		// Access outgoing edges of each vertex.
		boost::tie(vertexIt, vertexEnd) = boost::vertices(g);
		for (; vertexIt != vertexEnd; ++vertexIt)
		{
			std::cout << "out-edges for " << *vertexIt << ": ";
			boost::tie(outedgeIt, outedgeEnd) = boost::out_edges(*vertexIt, g);  // Similar to incoming edges.
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

			// NOTE [error] >> compile-time error.
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

		//
		// NOTE [error] >> compile-time error.
		//	- Haven't added any property using boost::vertex_name_t.
		//typename boost::property_map<map_type, boost::vertex_name_t>::type vertex_names = boost::get(boost::vertex_name, map);
		//typename boost::property_map<map_type, int>::type vertex_populations = boost::get(&City::population, map);  // NOTE [error] >> compile-time error.

		auto vertex_names = boost::get(&City::name, map);
		auto vertex_populations = boost::get(&City::population, map);

		std::cout << "type name of names = " << typeid(vertex_names).name() << std::endl;
		std::cout << "type name of populations = " << typeid(vertex_populations).name() << std::endl;

		// Output.
		boost::write_graphviz(
			std::cout, map,
			boost::make_label_writer(vertex_names),  // Vertex properties writer.
			//boost::make_label_writer(vertex_populations),  // Vertex properties writer.
			boost::make_label_writer(boost::get(&Highway::name, map))  // Edge properties writer.
		);
	}

	// Without bundled properties.
	{
#if defined(__USE_BOOST_PROPERTY_KIND)
		typedef boost::adjacency_list <
			boost::listS, boost::vecS, boost::bidirectionalS,
			// Vertex properties.
			boost::property<boost::vertex_name_t, std::string,
				boost::property<boost::my_population_type, int,
					boost::property<boost::my_zipcodes_type, std::vector<int> >
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
		typedef boost::vertex_color_t population_type;
		const population_type population = boost::vertex_color;
		typedef boost::vertex_index_t zipcodes_type;
		typedef boost::edge_color_t edge_speed_limit_type;
		const edge_speed_limit_type edge_speed_limit = boost::edge_color;
		typedef boost::edge_capacity_t edge_lanes_type;
		typedef boost::edge_flow_t edge_divided_type;
		//typedef boost::graph_visitor_t graph_use_right_type;
		enum graph_use_right_type { graph_use_right };
		enum graph_use_metric_type { graph_use_metric };

		typedef boost::adjacency_list <
			boost::listS, boost::vecS, boost::bidirectionalS,
			// Vertex properties.
			boost::property<boost::vertex_name_t, std::string,
				boost::property<population_type, int,
					boost::property<zipcodes_type, std::vector<int> >
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
		typename boost::property_map<map_type, boost::my_population_type>::type vertex_populations = boost::get(boost::my_population, map);
#else
		typename boost::property_map<map_type, boost::vertex_name_t>::type vertex_names = boost::get(boost::vertex_name, map);
		typename boost::property_map<map_type, population_type>::type vertex_populations = boost::get(population, map);
#endif
		boost::get(vertex_names, v) = "Troy";
		boost::get(vertex_populations, v) = 49170;

		map_type::edge_descriptor e = *boost::edges(map).first;
#if defined(__USE_BOOST_PROPERTY_KIND)
		typename boost::property_map<map_type, boost::my_edge_speed_limit_type>::type edge_speed_limits = boost::get(boost::my_edge_speed_limit, map);
#else
		typename boost::property_map<map_type, edge_speed_limit_type>::type edge_speed_limits = boost::get(edge_speed_limit, map);
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
