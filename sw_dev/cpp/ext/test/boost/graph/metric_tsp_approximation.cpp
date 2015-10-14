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
#include <fstream>
#include <set>
#include <ctime>

// Travelling salesman problem (TSP).
// Hamiltonian path/cycle problem.
// REF [file] >> ${BOOST_HOME}/libs/graph/test/metric_tsp_approx.cpp
int metric_tsp_approximation()
{
    typedef vector<simple_point<double> > PositionVec;
    typedef adjacency_matrix<undirectedS, no_property, property<edge_weight_t, double> > Graph;
    typedef graph_traits<Graph>::vertex_descriptor Vertex;
    typedef vector<Vertex> Container;
    typedef property_map<Graph, edge_weight_t>::type WeightMap;
    typedef property_map<Graph, vertex_index_t>::type VertexMap;

    // Make sure that the the we can parse the given file.
    if (argc < 2)
    {
        usage();
        // return -1;
        return 0;
    }

    // Open the graph file, failing if one isn't given on the command line.
    ifstream fin(argv[1]);
    if (!fin)
    {
        usage();
        // return -1;
        return 0;
    }

    string line;
    PositionVec position_vec;

    int n(0);
    while (getline(fin, line))
    {
        simple_point<double> vertex;

        size_t idx(line.find(","));
        string xStr(line.substr(0, idx));
        string yStr(line.substr(idx + 1, line.size() - idx));

        vertex.x = lexical_cast<double>(xStr);
        vertex.y = lexical_cast<double>(yStr);

        position_vec.push_back(vertex);
        ++n;
    }

    fin.close();

    Container c;
    Graph g(position_vec.size());
    WeightMap weight_map(get(edge_weight, g));
    VertexMap v_map = get(vertex_index, g);

    connectAllEuclidean(g, position_vec, weight_map, v_map, n);

    metric_tsp_approx_tour(g, back_inserter(c));

    for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
    {
        cout << *itr << " ";
    }
    cout << endl << endl;

    c.clear();

    checkAdjList(position_vec);

    metric_tsp_approx_from_vertex(
        g, *vertices(g).first, get(edge_weight, g), get(vertex_index, g),
        tsp_tour_visitor<back_insert_iterator<vector<Vertex> > >(back_inserter(c))
    );

    for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr)
    {
        cout << *itr << " ";
    }
    cout << endl << endl;

    c.clear();

    double len(0.0);
    try
    {
        metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
    }
    catch (const bad_graph &e)
    {
        cerr << "bad_graph: " << e.what() << endl;
        return -1;
    }

    cout << "Number of points: " << num_vertices(g) << endl;
    cout << "Number of edges: " << num_edges(g) << endl;
    cout << "Length of Tour: " << len << endl;

    int cnt(0);
    pair<Vertex,Vertex> triangleEdge;
    for (vector<Vertex>::iterator itr = c.begin(); itr != c.end(); ++itr, ++cnt)
    {
        cout << *itr << " ";

        if (cnt == 2)
        {
            triangleEdge.first = *itr;
        }
        if (cnt == 3)
        {
            triangleEdge.second = *itr;
        }
    }
    cout << endl << endl;
    c.clear();

    testScalability(1000);

    // if the graph is not fully connected then some of the assumed triangle-inequality edges may not exist.
    remove_edge(edge(triangleEdge.first, triangleEdge.second, g).first, g);

    // Make sure that we can actually trap incomplete graphs.
    bool caught = false;
    try
    {
        double len = 0.0;
        metric_tsp_approx(g, make_tsp_tour_len_visitor(g, back_inserter(c), len, weight_map));
    }
    catch (const bad_graph &e)
    {
        caught = true;
    }
    BOOST_ASSERT(caught);
}
