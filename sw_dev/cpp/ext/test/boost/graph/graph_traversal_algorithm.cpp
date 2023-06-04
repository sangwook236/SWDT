#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/astar_search.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/grid_graph.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/random.hpp>
#include <boost/pending/indirect_cmp.hpp>
#include <boost/range/irange.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random.hpp>
#include <boost/unordered_map.hpp>
#include <boost/unordered_set.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include <cmath>
#include <ctime>


namespace {
namespace local {

class custom_dfs_visitor : public boost::default_dfs_visitor
{
public:
	// This is invoked when a vertex is encountered for the first time.
	template<typename Vertex, typename Graph>
	void discover_vertex(const Vertex& u, const Graph&) const
	{ std::cout << "at " << u << std::endl; }

	// This is invoked on every outgoing edge_type from the vertex after the vertex is discovered.
	template<typename Edge, typename Graph>
	void examine_edge(const Edge& e, const Graph&) const
	{ std::cout << "examining edges " << e << std::endl; }

/*
	// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/graph/doc/depth_first_search.html

	// This is invoked on every vertex of the graph before the start of the graph search.
	template<typename Vertex, typename Graph>
	void initialize_vertex(Vertex u, Graph &g)
	{
		// Do something.
	}

	// This is invoked on the source vertex before traversal begins.
	// This invoked on the source vertex once before the start of the search..
	template<typename Vertex, typename Graph>
	void start_vertex(Vertex u, Graph &g)
	{
		// Do something.
	}

	// This is invoked when a vertex is invoked for the first time.
	// This is invoked when a vertex is encountered for the first time.
	template<typename Vertex, typename Graph>
	void discover_vertex(Vertex u, Graph &g)
	{
		// Do something.
	}

	// This is invoked on a vertex after all of its out edges have been added to the search tree and all of the adjacent vertices have been discovered (but before their out-edges have been examined).
	// If u is the root of a tree, finish_vertex is invoked after the same is invoked on all other elements of the tree.
	// If u is a leaf, then this method is invoked after all outgoing edges from u have been examined.
	template<typename Vertex, typename Graph>
	void finish_vertex(Vertex u, Graph &g)
	{
		// Do something.
	}

	// This is invoked on every outgoing edge of each vertex after it is discovered.
	template<typename Edge, typename Graph>
	void examine_edge(Edge e, Graph &g)
	{
		// Do something.
	}

	// This is invoked on an edge after it becomes a member of the edges that form the search tree.
	// This is invoked on each edge as it becomes a member of the edges that form the search tree. If you wish to record predecessors, do so at this event point.
	template<typename Edge, typename Graph>
	void tree_edge(Edge e, Graph &g)
	{
		// Do something.
	}

	// This is invoked on the back edges of a graph; used for an undirected graph, and because (u, v) and (v, u) are the same edges, both tree_edge and back_edge are invoked.
	template<typename Edge, typename Graph>
	void back_edge(Edge e, Graph &g)
	{
		// Do something.
	}

	// This is invoked on forward or cross edges in the graph. In an undirected graph this method is never called.
	template<typename Edge, typename Graph>
	void forward_or_cross_edge(Edge e, Graph &g)
	{
		// Do something.
	}

	// This is invoked on the non-tree edges in the graph as well as on each tree edge after its target vertex is finished.
	template<typename Edge, typename Graph>
	void finish_edge(Edge e, Graph &g)
	{
		// Do something.
	}
*/
};

void simple_dfs_example()
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
#if 0
	boost::depth_first_search(g, boost::visitor(vis).root_vertex(boost::vertex(0, g)));
#elif 0
	// TODO [check] >> This implementation isn't tested yet.
	std::vector<boost::default_color_type> vertex_colors(boost::num_vertices(g), boost::white_color);
	boost::depth_first_search(g, boost::visitor(vis), boost::make_iterator_property_map(vertex_colors.begin(), boost::get(boost::vertex_index, g)), boost::vertex(0, g));
#else
	boost::depth_first_search(g, boost::visitor(vis));
#endif
}

template<typename TimeMap>
class bfs_time_visitor : public boost::default_bfs_visitor
{
private:
	typedef typename boost::property_traits<TimeMap>::value_type T;

public:
	bfs_time_visitor(TimeMap tmap, T& t)
	: m_timemap(tmap), m_time(t)
	{}

public:
	// This is invoked the first time the algorithm encounters vertex u.
	// All vertices closer to the source vertex have been discovered, and vertices further from the source have not yet been discovered.
	template<typename Vertex, typename Graph>
	void discover_vertex(const Vertex& u, const Graph&) const
	{
		boost::put(m_timemap, u, m_time++);
	}

/*
	// REF [site] >> http://www.boost.org/doc/libs/1_61_0/libs/graph/doc/breadth_first_search.html

	// This is invoked on every vertex before the start of the search.
	template<typename Vertex, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden initialize_vertex(Vertex u, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked the first time the algorithm encounters vertex u. All vertices closer to the source vertex have been discovered, and vertices further from the source have not yet been discovered.
	template<typename Vertex, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden discover_vertex(Vertex u, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked in each vertex as it is removed from the queue.
	template<typename Vertex, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden examine_vertex(Vertex u, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked after all of the out edges of u have been examined and all of the adjacent vertices have been discovered.
	template<typename Vertex, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden finish_vertex(Vertex u, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked on every out-edge of each vertex immediately after the vertex is removed from the queue.
	template<typename Edge, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden examine_edge(Edge e, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked (in addition to examine_edge()) if the edge is a tree edge. The target vertex of edge e is discovered at this time.
	template<typename Edge, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden tree_edge(Edge e, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked (in addition to examine_edge()) if the edge is not a tree edge.
	template<typename Edge, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden non_tree_edge(Edge e, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked (in addition to non_tree_edge()) if the target vertex is colored gray at the time of examination. The color gray indicates that the vertex is currently in the queue.
	template<typename Edge, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden gray_target(Edge e, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}

	// This is invoked (in addition to non_tree_edge()) if the target vertex is colored black at the time of examination. The color black indicates that the vertex is no longer in the queue.
	template<typename Edge, typename Graph>
	boost::graph::bfs_visitor_event_not_overridden black_target(Edge e, Graph &g)
	{
		// Do something.
		return boost::graph::bfs_visitor_event_not_overridden();
	}
*/

private:
	TimeMap m_timemap;
	T& m_time;
};

// REF [file] >> ${BOOST_HOME}/libs/graph/example/bfs-example.cpp
void bfs_example()
{
	// Select the graph type we wish to use.
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> graph_type;

	// Set up the vertex IDs and names.
	enum { r, s, t, u, v, w, x, y, N };
	const char *name = "rstuvwxy";
	// Specify the edges in the graph.
	typedef std::pair<int, int> edge_type;
	edge_type edge_array[] = {
		edge_type(r, s), edge_type(r, v), edge_type(s, w), edge_type(w, r), edge_type(w, t),
		edge_type(w, x), edge_type(x, t), edge_type(t, u), edge_type(x, y), edge_type(u, y)
	};

	// Create the graph object.
	const int n_edges = sizeof(edge_array) / sizeof(edge_type);
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
	// VC++ has trouble with the edge iterator constructor.
	graph_type g(N);
	for (std::size_t j = 0; j < n_edges; ++j)
		boost::add_edge(edge_array[j].first, edge_array[j].second, g);
#else
	typedef boost::graph_traits<graph_type>::vertices_size_type v_size_type;
	graph_type g(edge_array, edge_array + n_edges, v_size_type(N));
#endif

	// Typedefs.
	typedef boost::graph_traits<graph_type>::vertices_size_type size_type;

	// A vector to hold the discover time property for each vertex.
	std::vector<size_type> dtime(boost::num_vertices(g));
	typedef boost::iterator_property_map<std::vector<size_type>::iterator, boost::property_map<graph_type, boost::vertex_index_t>::const_type> dtime_pm_type;
	dtime_pm_type dtime_pm(dtime.begin(), boost::get(boost::vertex_index, g));

	size_type time = 0;
	bfs_time_visitor<dtime_pm_type> vis(dtime_pm, time);
	boost::breadth_first_search(g, boost::vertex(s, g), boost::visitor(vis));

	// Use std::sort to order the vertices by their discover time.
	std::vector<boost::graph_traits<graph_type>::vertices_size_type> discover_order(N);
	boost::integer_range<int> range(0, N);
	std::copy(range.begin(), range.end(), discover_order.begin());
	std::sort(discover_order.begin(), discover_order.end(), boost::indirect_cmp<dtime_pm_type, std::less<size_type> >(dtime_pm));

	std::cout << "Order of discovery: ";
	for (int i = 0; i < N; ++i)
		std::cout << name[discover_order[i]] << " ";
	std::cout << std::endl;
}

template<typename Graph, typename VertexNameMap, typename TransDelayMap>
void build_router_network(Graph& g, VertexNameMap name_map, TransDelayMap delay_map)
{
	typename boost::graph_traits<Graph>::vertex_descriptor a, b, c, d, e;
	a = boost::add_vertex(g);
	name_map[a] = 'a';
	b = boost::add_vertex(g);
	name_map[b] = 'b';
	c = boost::add_vertex(g);
	name_map[c] = 'c';
	d = boost::add_vertex(g);
	name_map[d] = 'd';
	e = boost::add_vertex(g);
	name_map[e] = 'e';

	typename boost::graph_traits<Graph>::edge_descriptor ed;
	bool inserted;

	boost::tie(ed, inserted) = boost::add_edge(a, b, g);
	delay_map[ed] = 1.2;
	boost::tie(ed, inserted) = boost::add_edge(a, d, g);
	delay_map[ed] = 4.5;
	boost::tie(ed, inserted) = boost::add_edge(b, d, g);
	delay_map[ed] = 1.8;
	boost::tie(ed, inserted) = boost::add_edge(c, a, g);
	delay_map[ed] = 2.6;
	boost::tie(ed, inserted) = boost::add_edge(c, e, g);
	delay_map[ed] = 5.2;
	boost::tie(ed, inserted) = boost::add_edge(d, c, g);
	delay_map[ed] = 0.4;
	boost::tie(ed, inserted) = boost::add_edge(d, e, g);
	delay_map[ed] = 3.3;
}

template<typename VertexNameMap>
class bfs_name_printer : public boost::default_bfs_visitor
{
// Inherit default (empty) event point actions.
public:
	bfs_name_printer(VertexNameMap n_map)
	: m_name_map(n_map)
	{}

public:
	template<typename Vertex, typename Graph>
	void discover_vertex(Vertex u, const Graph &) const
	{
		std::cout << boost::get(m_name_map, u) << ' ';
	}

private:
	VertexNameMap m_name_map;
};

struct VP
{
	char name;
};

struct EP
{
	double weight;
};

// REF [file] >> ${BOOST_HOME}/libs/graph/example/bfs-name-printer.cpp
void bfs_name_printer_example()
{
	typedef boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, VP, EP> graph_type;
	graph_type g;

	boost::property_map<graph_type, char VP::*>::type name_map = boost::get(&VP::name, g);
	boost::property_map<graph_type, double EP::*>::type delay_map = boost::get(&EP::weight, g);

	build_router_network(g, name_map, delay_map);

	typedef boost::property_map<graph_type, char VP::*>::type VertexNameMap;
	boost::graph_traits<graph_type>::vertex_descriptor a = *boost::vertices(g).first;
	bfs_name_printer<VertexNameMap> vis(name_map);

	std::cout << "BFS vertex discover order: ";
	boost::breadth_first_search(g, a, boost::visitor(vis));
	std::cout << std::endl;
}

// Auxiliary types.
struct location_type
{
	float y, x;  // lat, long.
};

typedef float cost_type;

template <class Name, class LocMap>
class city_writer
{
public:
	city_writer(Name n, LocMap l, float _minx, float _maxx, float _miny, float _maxy, unsigned int _ptx, unsigned int _pty)
	: name(n), loc(l), minx(_minx), maxx(_maxx), miny(_miny), maxy(_maxy), ptx(_ptx), pty(_pty)
	{}

public:
	template<class Vertex>
	void operator()(std::ostream& out, const Vertex& v) const
	{
		const float px = 1 - (loc[v].x - minx) / (maxx - minx);
		const float py = (loc[v].y - miny) / (maxy - miny);
		out << "[label=\"" << name[v] << "\", pos=\""
			<< static_cast<unsigned int>(ptx * px) << ","
			<< static_cast<unsigned int>(pty * py)
			<< "\", fontsize=\"11\"]";
	}

private:
	Name name;
	LocMap loc;
	float minx, maxx, miny, maxy;
	unsigned int ptx, pty;
};

template <class WeightMap>
class time_writer
{
public:
	time_writer(WeightMap w)
	: wm(w)
	{}

public:
	template <class Edge>
	void operator()(std::ostream &out, const Edge& e) const
	{
		out << "[label=\"" << wm[e] << "\", fontsize=\"11\"]";
	}
private:
	WeightMap wm;
};

// Euclidean distance heuristic.
template<class Graph, class CostType, class LocMap>
class distance_heuristic : public boost::astar_heuristic<Graph, CostType>
{
public:
	typedef typename boost::graph_traits<Graph>::vertex_descriptor Vertex;

public:
	distance_heuristic(LocMap l, Vertex goal)
	: m_location(l), m_goal(goal)
	{}

public:
	CostType operator()(Vertex u) const
	{
		const CostType dx = m_location[m_goal].x - m_location[u].x;
		const CostType dy = m_location[m_goal].y - m_location[u].y;
		return std::sqrt(dx * dx + dy * dy);
	}

private:
	LocMap m_location;
	Vertex m_goal;
};

struct found_goal {};  // Exception for termination.

// Visitor that terminates when we find the goal.
template<class Vertex>
class astar_goal_visitor : public boost::default_astar_visitor
{
public:
	astar_goal_visitor(Vertex goal)
	: m_goal(goal)
	{}

public:
	template<class Graph>
	void examine_vertex(Vertex u, Graph& g)
	{
		if (u == m_goal)
			throw found_goal();
	}

private:
	Vertex m_goal;
};

// REF [file] >> ${BOOST_HOME}/libs/graph/example/astar-cities.cpp
void astar_cities_example()
{
	// Specify some types.
	typedef boost::adjacency_list<boost::listS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, cost_type> > graph_type;
	typedef boost::property_map<graph_type, boost::edge_weight_t>::type weight_map_type;
	typedef graph_type::vertex_descriptor vertex_type;
	typedef graph_type::edge_descriptor edge_descriptor_type;
	typedef std::pair<int, int> edge_type;

	// Specify data.
	enum nodes {
		Troy, LakePlacid, Plattsburgh, Massena, Watertown, Utica,
		Syracuse, Rochester, Buffalo, Ithaca, Binghamton, Woodstock,
		NewYork, N
	};
	const char *name[] = {
		"Troy", "Lake Placid", "Plattsburgh", "Massena",
		"Watertown", "Utica", "Syracuse", "Rochester", "Buffalo",
		"Ithaca", "Binghamton", "Woodstock", "New York"
	};
	location_type locations[] = {  // lat/long.
		{42.73, 73.68}, {44.28, 73.99}, {44.70, 73.46},
		{44.93, 74.89}, {43.97, 75.91}, {43.10, 75.23},
		{43.04, 76.14}, {43.17, 77.61}, {42.89, 78.86},
		{42.44, 76.50}, {42.10, 75.91}, {42.04, 74.11},
		{40.67, 73.94}
	};
	edge_type edge_array[] = {
		edge_type(Troy, Utica), edge_type(Troy, LakePlacid),
		edge_type(Troy, Plattsburgh), edge_type(LakePlacid, Plattsburgh),
		edge_type(Plattsburgh, Massena), edge_type(LakePlacid, Massena),
		edge_type(Massena, Watertown), edge_type(Watertown, Utica),
		edge_type(Watertown, Syracuse), edge_type(Utica, Syracuse),
		edge_type(Syracuse, Rochester), edge_type(Rochester, Buffalo),
		edge_type(Syracuse, Ithaca), edge_type(Ithaca, Binghamton),
		edge_type(Ithaca, Rochester), edge_type(Binghamton, Troy),
		edge_type(Binghamton, Woodstock), edge_type(Binghamton, NewYork),
		edge_type(Syracuse, Binghamton), edge_type(Woodstock, Troy),
		edge_type(Woodstock, NewYork)
	};
	const unsigned int num_edges = sizeof(edge_array) / sizeof(edge_type);
	cost_type weights[] = {  // estimated travel time [mins].
		96, 134, 143, 65, 115, 133, 117, 116, 74, 56,
		84, 73, 69, 70, 116, 147, 173, 183, 74, 71, 124
	};

	// Create graph.
	graph_type g(N);
	weight_map_type weightmap = boost::get(boost::edge_weight, g);
	for (std::size_t j = 0; j < num_edges; ++j)
	{
		edge_descriptor_type e;
		bool inserted;
		boost::tie(e, inserted) = boost::add_edge(edge_array[j].first, edge_array[j].second, g);
		weightmap[e] = weights[j];
	}

	// Pick random start/goal.
	boost::mt19937 gen(std::time(0));
	vertex_type start = boost::random_vertex(g, gen);
	vertex_type goal = boost::random_vertex(g, gen);

	std::cout << "Start vertex: " << name[start] << std::endl;
	std::cout << "Goal vertex: " << name[goal] << std::endl;

	std::ofstream dotfile;
	dotfile.open("./data/boost/astar_cities.dot");
	boost::write_graphviz(
		dotfile,
		g,
		city_writer<const char**, location_type*>(name, locations, 73.46, 78.86, 40.67, 44.93, 480, 400),
		time_writer<weight_map_type>(weightmap)
	);

	std::vector<graph_type::vertex_descriptor> p(boost::num_vertices(g));
	std::vector<cost_type> d(boost::num_vertices(g));
	try
	{
		// Call astar named parameter interface.
		boost::astar_search_tree(
			g,
			start,
			distance_heuristic<graph_type, cost_type, location_type*>(locations, goal),
			boost::predecessor_map(boost::make_iterator_property_map(p.begin(), boost::get(boost::vertex_index, g))).
				distance_map(boost::make_iterator_property_map(d.begin(), boost::get(boost::vertex_index, g))).
					visitor(astar_goal_visitor<vertex_type>(goal))
		);
	}
	catch (const found_goal &)  // Found a path to the goal.
	{
		std::list<vertex_type> shortest_path;
		for (vertex_type v = goal; ; v = p[v])
		{
			shortest_path.push_front(v);
			if (p[v] == v)
				break;
		}
		std::cout << "Shortest path from " << name[start] << " to " << name[goal] << ": ";
		std::list<vertex_type>::iterator spi = shortest_path.begin();
		std::cout << name[start];
		for (++spi; spi != shortest_path.end(); ++spi)
			std::cout << " -> " << name[*spi];
		std::cout << std::endl << "Total travel time: " << d[goal] << std::endl;
		return;
	}

	std::cout << "Didn't find a path from " << name[start] << "to" << name[goal] << "!" << std::endl;
}

class Maze
{
public:
	static const std::size_t GRID_RANK = 2;

	// Distance traveled in the Maze.
	typedef double distance_type;

	typedef boost::grid_graph<GRID_RANK> grid_type;
	typedef boost::graph_traits<grid_type>::vertex_descriptor vertex_type;
	typedef boost::graph_traits<grid_type>::vertices_size_type vertices_size_type;

	// A hash function for vertices.
	struct vertex_hash : std::unary_function<vertex_type, std::size_t>
	{
		std::size_t operator()(vertex_type const& u) const
		{
			std::size_t seed = 0;
			boost::hash_combine(seed, u[0]);
			boost::hash_combine(seed, u[1]);
			return seed;
		}
	};

	typedef boost::unordered_set<vertex_type, vertex_hash> vertex_set_type;
	typedef boost::vertex_subset_complement_filter<grid_type, vertex_set_type>::type filtered_grid_type;

	// Euclidean heuristic for a grid.
	// This calculates the Euclidean distance between a vertex and a goal vertex.
	class euclidean_heuristic : public boost::astar_heuristic<filtered_grid_type, double>
	{
	public:
		euclidean_heuristic(vertex_type goal)
		: m_goal(goal)
		{}

		double operator()(vertex_type v)
		{
			return std::sqrt(std::pow(double(m_goal[0] - v[0]), 2) + std::pow(double(m_goal[1] - v[1]), 2));
		}

	private:
		vertex_type m_goal;
	};

	// Visitor that terminates when we find the goal vertex.
	struct astar_goal_visitor : public boost::default_astar_visitor
	{
	public:
		astar_goal_visitor(vertex_type goal)
		: m_goal(goal)
		{};

	public:
		void examine_vertex(vertex_type u, const filtered_grid_type&)
		{
			if (u == m_goal)
				throw found_goal();
		}

	private:
		vertex_type m_goal;
	};

public:
	friend std::ostream& operator<<(std::ostream&, const Maze&);

public:
	Maze()
	: m_grid(create_grid(0, 0)), m_barrier_grid(create_barrier_grid())
	{}
	Maze(std::size_t x, std::size_t y)
	: m_grid(create_grid(x, y)), m_barrier_grid(create_barrier_grid())
	{}

public:
	// The length of the Maze along the specified dimension.
	vertices_size_type length(std::size_t d) const { return m_grid.length(d); }

	bool has_barrier(vertex_type u) const
	{ return m_barriers.find(u) != m_barriers.end(); }

	// Try to find a path from the lower-left-hand corner source (0,0) to the upper-right-hand corner goal (x-1, y-1).
	vertex_type source() const { return vertex(0, m_grid); }
	vertex_type goal() const
	{ return vertex(num_vertices(m_grid) - 1, m_grid); }

	bool solve();
	bool solved() const { return !m_solution.empty(); }
	bool solution_contains(vertex_type u) const
	{ return m_solution.find(u) != m_solution.end(); }

	grid_type& grid() { return m_grid; }
	const grid_type& grid() const { return m_grid; }

	vertex_set_type& barriers() { return m_barriers; }
	const vertex_set_type& barriers() const { return m_barriers; }

private:
	// Create the underlying rank-2 grid with the specified dimensions.
	grid_type create_grid(std::size_t x, std::size_t y)
	{
		boost::array<std::size_t, GRID_RANK> lengths = {{ x, y }};
		return grid_type(lengths);
	}

	// Filter the barrier vertices out of the underlying grid.
	filtered_grid_type create_barrier_grid()
	{
		return boost::make_vertex_subset_complement_filter(m_grid, m_barriers);
	}

private:
	// The grid underlying the Maze
	grid_type m_grid;
	// The underlying Maze grid with barrier vertices filtered out
	filtered_grid_type m_barrier_grid;
	// The barriers in the Maze
	vertex_set_type m_barriers;
	// The vertices on a solution path through the Maze
	vertex_set_type m_solution;
	// The length of the solution path
	distance_type m_solution_length;
};

// Solve the Maze using A-star search.
// Return true if a solution was found.
bool Maze::solve()
{
	boost::static_property_map<distance_type> weight(1);
	// The predecessor map is a vertex-to-vertex mapping.
	typedef boost::unordered_map<vertex_type, vertex_type, vertex_hash> pred_map_type;
	pred_map_type predecessor;
	boost::associative_property_map<pred_map_type> pred_pmap(predecessor);
	// The distance map is a vertex-to-distance mapping.
	typedef boost::unordered_map<vertex_type, distance_type, vertex_hash> dist_map_type;
	dist_map_type distance;
	boost::associative_property_map<dist_map_type> dist_pmap(distance);

	vertex_type s = source();
	vertex_type g = goal();
	euclidean_heuristic heuristic(g);
	astar_goal_visitor visitor(g);

	try
	{
		boost::astar_search(
			m_barrier_grid,
			s,
			heuristic,
			boost::weight_map(weight).
				predecessor_map(pred_pmap).
					distance_map(dist_pmap).
						visitor(visitor)
		);
	}
	catch(const found_goal &)
	{
		// Walk backwards from the goal through the predecessor chain adding vertices to the solution path.
		for (vertex_type u = g; u != s; u = predecessor[u])
			m_solution.insert(u);
		m_solution.insert(s);
		m_solution_length = distance[g];
		return true;
	}

	return false;
}

#define BARRIER "#"

// Print the Maze as an ASCII map.
std::ostream& operator<<(std::ostream& output, const Maze& maze)
{
	// Header
	for (Maze::vertices_size_type i = 0; i < maze.length(0) + 2; ++i)
		output << BARRIER;
	output << std::endl;
	// Body
	for (int y = maze.length(1) - 1; y >= 0; --y)
	{
		// Enumerate rows in reverse order and columns in regular order so that
		// (0,0) appears in the lower left-hand corner.  This requires that y be
		// int and not the unsigned vertices_size_type because the loop exit
		// condition is y==-1.
		for (Maze::vertices_size_type x = 0; x < maze.length(0); ++x)
		{
			// Put a barrier on the left-hand side.
			if (x == 0)
				output << BARRIER;
			// Put the character representing this point in the Maze grid.
			Maze::vertex_type u = {{ x, Maze::vertices_size_type(y) }};
			if (maze.solution_contains(u))
				output << ".";
			else if (maze.has_barrier(u))
				output << BARRIER;
			else
				output << " ";
			// Put a barrier on the right-hand side.
			if (x == maze.length(0) - 1)
				output << BARRIER;
		}
		// Put a newline after every row except the last one.
		output << std::endl;
	}
	// Footer
	for (Maze::vertices_size_type i = 0; i < maze.length(0) + 2; ++i)
		output << BARRIER;
	if (maze.solved())
		output << std::endl << "Solution length " << maze.m_solution_length;
	return output;
}

// Return a random integer in the interval [a, b].
std::size_t random_int(std::size_t a, std::size_t b, boost::mt19937 &random_generator)
{
	if (b < a)
		b = a;
	boost::uniform_int<> dist(a, b);
	boost::variate_generator<boost::mt19937&, boost::uniform_int<> > generate(random_generator, dist);
	return generate();
}

// REF [file] >> ${BOOST_HOME}/libs/graph/example/astar_maze.cpp
void astar_maze_example()
{
	const std::size_t x = 20;
	const std::size_t y = 10;

	boost::mt19937 random_generator;
	random_generator.seed(std::time(0));
	Maze maze(x, y);
	{
		Maze::vertices_size_type n = num_vertices(maze.grid());
		Maze::vertex_type s = maze.source();
		Maze::vertex_type g = maze.goal();

		// One quarter of the cells in the Maze should be barriers.
		int barriers = n / 4;
		while (barriers > 0)
		{
			// Choose horizontal or vertical direction.
			const std::size_t direction = random_int(0, 1, random_generator);
			// Walls range up to one quarter the dimension length in this direction.
			Maze::vertices_size_type wall = random_int(1, maze.length(direction) / 4, random_generator);
			// Create the wall while decrementing the total barrier count.
			Maze::vertex_type u = vertex(random_int(0, n - 1, random_generator), maze.grid());
			while (wall)
			{
				// Start and goal spaces should never be barriers.
				if (u != s && u != g)
				{
					--wall;
					if (!maze.has_barrier(u))
					{
						maze.barriers().insert(u);
						barriers--;
					}
				}
				Maze::vertex_type v = maze.grid().next(u, direction);
				// Stop creating this wall if we reached the Maze's edge.
				if (u == v) break;
				u = v;
			}
		}
	}

	if (maze.solve())
	{
		std::cout << "Solved the Maze." << std::endl;
		std::cout << maze << std::endl;
	}
	else
		std::cout << "The Maze is not solvable." << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace boost_graph {

// REF [site] >> http://www.ibm.com/developerworks/aix/library/au-aix-boost-graph/index.html
void traversal()
{
	std::cout << "Depth-first search -------------------------------------------" << std::endl;
	local::simple_dfs_example();

	std::cout << "\nBreadth-first search -----------------------------------------" << std::endl;
	local::bfs_example();
	local::bfs_name_printer_example();

	std::cout << "\nA* search ----------------------------------------------------" << std::endl;
	local::astar_cities_example();
	local::astar_maze_example();
}

}  // namespace boost_graph
