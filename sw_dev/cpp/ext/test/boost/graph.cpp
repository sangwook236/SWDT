#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graphviz.hpp>
#include <algorithm>
#include <utility>


namespace {
namespace local {

template <class Graph>
struct exercise_vertex
{
	exercise_vertex(Graph &g, const char name[])
	: g_(g), name_(name)
	{}

	typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

	void operator()(const Vertex &v) const
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
			Vertex src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_id, src)] << "," << name_[boost::get(vertex_id, targ)] << ") ";
		}
		std::cout << std::endl;

		// write out the incoming edges
		std::cout << "\tin-edges: ";
		typename boost::graph_traits<Graph>::in_edge_iterator in_i, in_end;
		for (boost::tie(in_i, in_end) = boost::in_edges(v, g_); in_i != in_end; ++in_i)
		{
			e = *in_i;
			Vertex src = boost::source(e, g_), targ = boost::target(e, g_);
			std::cout << "(" << name_[boost::get(vertex_id, src)] << "," << name_[boost::get(vertex_id, targ)] << ") ";
		}
		std::cout << std::endl;

		// write out all adjacent vertices
		std::cout << "\tadjacent vertices: ";
		typename boost::graph_traits<Graph>::adjacency_iterator ai, ai_end;
		for (boost::tie(ai, ai_end) = boost::adjacent_vertices(v, g_);  ai != ai_end; ++ai)
			std::cout << name_[boost::get(vertex_id, *ai)] <<  " ";
		std::cout << std::endl;
	}

	Graph &g_;
	const char *name_;
};

template <class PredecessorMap>
class record_predecessors: public boost::dijkstra_visitor<>
{
public:
	record_predecessors(PredecessorMap p)
	: predecessor_(p)
	{}

	template <class Edge, class Graph>
	void edge_relaxed(Edge e, Graph &g)
	{
		// set the parent of the target(e) to source(e)
		boost::put(predecessor_, boost::target(e, g), boost::source(e, g));
	}

protected:
	PredecessorMap predecessor_;
};

template <class PredecessorMap>
record_predecessors<PredecessorMap> make_predecessor_recorder(PredecessorMap p)
{
	return record_predecessors<PredecessorMap>(p);
}

void boost_quick_tour()
{
	// create a typedef for the Graph type
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property<boost::edge_weight_t, float> > Graph;

	// Make convenient labels for the vertices
	enum { A, B, C, D, E, N };
	const int num_vertices = N;
	const char name[] = "ABCDE";

	// writing out the edges in the graph
	typedef std::pair<int, int> Edge;
	Edge edge_array[] = { Edge(A,B), Edge(A,D), Edge(C,A), Edge(D,C), Edge(C,E), Edge(B,D), Edge(D,E), };
	const int num_edges = sizeof(edge_array) / sizeof(edge_array[0]);

	// average transmission delay (in milliseconds) for each connection
	const float transmission_delay[] = { 1.2f, 4.5f, 2.6f, 0.4f, 5.2f, 1.8f, 3.3f, 9.1f };
	
	// declare a graph object, adding the edges and edge properties
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// VC++ can't handle the iterator constructor
	Graph g(num_vertices);
	boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(edge_weight, g);
	for (std::size_t j = 0; j < num_edges; ++j)
	{
		boost::graph_traits<Graph>::edge_descriptor e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = transmission_delay[j];
	}
#else
	Graph g(edge_array, edge_array + num_edges, transmission_delay, num_vertices);
#endif

	const boost::property_map<Graph, boost::vertex_index_t>::type &vertex_id = boost::get(boost::vertex_index, g);
	const boost::property_map<Graph, boost::edge_weight_t>::type &trans_delay = boost::get(boost::edge_weight, g);

	std::cout << "----------------------------------------------------------" << std::endl;
	// output info
	{
		std::cout << "vertices(g) = ";
		typedef boost::graph_traits<Graph>::vertex_iterator vertex_iter;
		for (std::pair<vertex_iter, vertex_iter> vp = boost::vertices(g); vp.first != vp.second; ++vp.first)
			std::cout << name[boost::get(vertex_id, *vp.first)] <<  " ";
		std::cout << std::endl;

		std::cout << "edges(g) = ";
		boost::graph_traits<Graph>::edge_iterator ei, ei_end;
		for (boost::tie(ei, ei_end) = boost::edges(g); ei != ei_end; ++ei)
			std::cout << "(" << name[boost::get(vertex_id, boost::source(*ei, g))] << "," << name[boost::get(vertex_id, boost::target(*ei, g))] << ") ";
		std::cout << std::endl;
	}

	// output info
	{
		std::for_each(boost::vertices(g).first, boost::vertices(g).second, exercise_vertex<Graph>(g, name));
	}

	std::cout << "\n----------------------------------------------------------" << std::endl;
	// write a graph using graphviz
	{
		std::map<std::string, std::string> graph_attr, vertex_attr, edge_attr;
		graph_attr["size"] = "3,3";
		graph_attr["rankdir"] = "LR";
		graph_attr["ratio"] = "fill";
		vertex_attr["shape"] = "circle";

		boost::write_graphviz(
			std::cout,
			g,
			boost::make_label_writer(name),
			boost::make_label_writer(trans_delay),
			boost::make_graph_attributes_writer(graph_attr, vertex_attr, edge_attr)
		);
	}

	std::cout << "\n----------------------------------------------------------" << std::endl;
	// Dijkstra's algorithm
	{
		typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

		// vector for storing distance property
		std::vector<float> d(boost::num_vertices(g));

		// get the first vertex
		const Vertex s = *(boost::vertices(g).first);

		// invoke variant 2 of Dijkstra's algorithm
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]));

		std::cout << "distances from start vertex:" << std::endl;
		for (boost::graph_traits<Graph>::vertex_iterator vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
			std::cout << "\tdistance(" << vertex_id(*vi) << ") = " << d[*vi] << std::endl;
		std::cout << std::endl;
	}

	std::cout << "\n----------------------------------------------------------" << std::endl;
	//
	{
		typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;

		// vector for storing distance property
		std::vector<float> d(boost::num_vertices(g));

		// get the first vertex
		const Vertex s = *(boost::vertices(g).first);

		std::vector<Vertex> p(boost::num_vertices(g), boost::graph_traits<Graph>::null_vertex());  // the predecessor array
		boost::dijkstra_shortest_paths(g, s, boost::distance_map(&d[0]).visitor(make_predecessor_recorder(&p[0])));

		std::cout << "parents in the tree of shortest paths:" << std::endl;
		for (boost::graph_traits<Graph>::vertex_iterator vi = boost::vertices(g).first; vi != boost::vertices(g).second; ++vi)
		{
			std::cout << "\tparent(" << *vi;
			if (p[*vi] == boost::graph_traits<Graph>::null_vertex())
				std::cout << ") = no parent" << std::endl;
			else
				std::cout << ") = " << p[*vi] << std::endl;
		}
	}
}

}  // local
}  // unnamed namespace

void graph()
{
	local::boost_quick_tour();
}
