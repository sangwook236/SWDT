#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <boost/integer_traits.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/simple_point.hpp>
#include <boost/graph/metric_tsp_approx.hpp>
#include <boost/graph/graphviz.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <ctime>


namespace {
namespace local {

// add edges to the graph (for each node connect it to all other nodes).
template<typename VertexListGraph, typename PointContainer, typename WeightMap, typename VertexIndexMap>
void connectAllEuclidean(VertexListGraph &g, const PointContainer &points, WeightMap wmap, VertexIndexMap vmap, int /*sz*/)
{
	typedef typename boost::graph_traits<VertexListGraph>::edge_descriptor edge_type;
	typedef typename boost::graph_traits<VertexListGraph>::vertex_iterator vertex_iterator_type;

	edge_type e;
	bool inserted;

	std::pair<vertex_iterator_type, vertex_iterator_type> verts(boost::vertices(g));
	for (vertex_iterator_type src(verts.first); src != verts.second; ++src)
	{
		for (vertex_iterator_type dest(src); dest != verts.second; ++dest)
		{
			if (dest != src)
			{
				const double weight(std::sqrt(std::pow(static_cast<double>(points[vmap[*src]].x - points[vmap[*dest]].x), 2.0) +
					std::pow(static_cast<double>(points[vmap[*dest]].y - points[vmap[*src]].y), 2.0)));

				boost::tie(e, inserted) = boost::add_edge(*src, *dest, g);

				wmap[e] = weight;
			}
		}
	}
}

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

// Travelling salesman problem (TSP).
// Hamiltonian path/cycle problem.
// REF [file] >> ${BOOST_HOME}/libs/graph/test/metric_tsp_approx.cpp
void metric_tsp_approximation()
{
	typedef std::vector<boost::simple_point<double> > position_vector_type;
	typedef boost::adjacency_matrix<boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, double> > graph_type;
	typedef boost::graph_traits<graph_type>::vertex_descriptor vertex_type;
	typedef std::vector<vertex_type> container_type;
	typedef boost::property_map<graph_type, boost::edge_weight_t>::type weight_map_type;
	typedef boost::property_map<graph_type, boost::vertex_index_t>::type vertex_map_type;

	position_vector_type position_vec;
#if 1
	const int n(8);
	{
		boost::simple_point<double> vertex;

		vertex.x = 2; vertex.y = 5;
		position_vec.push_back(vertex);
		vertex.x = 2; vertex.y = 3;
		position_vec.push_back(vertex);
		vertex.x = 1; vertex.y = 2;
		position_vec.push_back(vertex);
		vertex.x = 4; vertex.y = 5;
		position_vec.push_back(vertex);
		vertex.x = 5; vertex.y = 4;
		position_vec.push_back(vertex);
		vertex.x = 4; vertex.y = 3;
		position_vec.push_back(vertex);
		vertex.x = 6; vertex.y = 3;
		position_vec.push_back(vertex);
		vertex.x = 3; vertex.y = 1;
		position_vec.push_back(vertex);
	}
#endif

	graph_type g(position_vec.size());
	weight_map_type weight_map(boost::get(boost::edge_weight, g));
	vertex_map_type v_map(boost::get(boost::vertex_index, g));

	// build a complete graph.
	local::connectAllEuclidean(g, position_vec, weight_map, v_map, n);

	//
	{
		container_type c;
		boost::metric_tsp_approx_tour(g, std::back_inserter(c));

		// Display.
		for (std::vector<vertex_type>::iterator itr = c.begin(); itr != c.end(); ++itr)
		{
			std::cout << *itr << " ";
		}
		std::cout << std::endl << std::endl;
	}

	//
	{
		//const vertex_type start_vertext(*boost::vertices(g).first);
		const vertex_type start_vertext(*(++boost::vertices(g).first));

		container_type c;
		boost::metric_tsp_approx_from_vertex(
			g,
			start_vertext,
			boost::get(boost::edge_weight, g),
			boost::get(boost::vertex_index, g),
			boost::tsp_tour_visitor<std::back_insert_iterator<std::vector<vertex_type> > >(std::back_inserter(c))
		);

		// Display.
		for (std::vector<vertex_type>::iterator itr = c.begin(); itr != c.end(); ++itr)
		{
			std::cout << *itr << " ";
		}
		std::cout << std::endl << std::endl;
	}

	{
		double len(0.0);
		container_type c;
		try
		{
			boost::metric_tsp_approx(g, boost::make_tsp_tour_len_visitor(g, std::back_inserter(c), len, weight_map));
		}
		catch (const boost::bad_graph &e)
		{
			std::cerr << "boost::bad_graph: " << e.what() << std::endl;
			return;
		}

		// Display.
		for (std::vector<vertex_type>::iterator itr = c.begin(); itr != c.end(); ++itr)
		{
			std::cout << *itr << " ";
		}
		std::cout << std::endl;

		std::cout << "Number of points: " << boost::num_vertices(g) << std::endl;
		std::cout << "Number of edges: " << boost::num_edges(g) << std::endl;
		std::cout << "Length of tour: " << len << std::endl;
	}
}

}  // namespace boost_graph
