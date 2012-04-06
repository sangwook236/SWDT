//#include "stdafx.h"
#include <mrpt/core.h>
#include <map>


namespace {
namespace local {

// adds a new edge to the graph. The edge is annotated with the relative position of the two nodes
void add_edge(const size_t from, const size_t to, const std::map<mrpt::math::CDijkstra<mrpt::poses::CPosePDFGaussian>::TNodeID, mrpt::poses::CPose2D> &real_poses, mrpt::poses::CNetworkOfPoses2D &graph_links, const mrpt::math::CMatrixDouble33 &cov)
{
	const mrpt::poses::CPose2D delta_p = real_poses.find(to)->second - real_poses.find(from)->second;
	graph_links.insertEdge(from, to, mrpt::poses::CPosePDFGaussian(delta_p, cov));
}
 
// weight is the distance between two nodes.
double get_Dijkstra_weight(const mrpt::poses::CPosePDFGaussian &edge)
{
	return edge.mean.norm();
}

}  // namespace local
}  // unnamed namespace

void dijkstra()
{
	mrpt::utils::CTicTac tictac;
	mrpt::poses::CNetworkOfPoses2D graph_links;
	mrpt::poses::CNetworkOfPoses2D::type_global_poses optimal_poses, optimal_poses_dijkstra;
	std::map<mrpt::math::CDijkstra<mrpt::poses::CPosePDFGaussian>::TNodeID, mrpt::poses::CPose2D> real_poses;
 
	mrpt::random::randomGenerator.randomize(444);
 
	// create a random graph:
	mrpt::math::CMatrixDouble33 cov;
	cov.unit();
	cov *= mrpt::utils::square(0.1);
 
	const size_t N_VERTEX = 15;
	const double DIST_THRES = 15;
	const double NODES_XY_MAX = 20;
 
	mrpt::vector_float xs, ys;
 
	for (size_t j = 0; j < N_VERTEX; ++j)
	{
		mrpt::poses::CPose2D p(
			mrpt::random::randomGenerator.drawUniform(-NODES_XY_MAX, NODES_XY_MAX),
			mrpt::random::randomGenerator.drawUniform(-NODES_XY_MAX, NODES_XY_MAX),
			mrpt::random::randomGenerator.drawUniform(-M_PI, M_PI)
		);
		real_poses[j] = p;
 
		// for the figure:
		xs.push_back(p.x());
		ys.push_back(p.y());
	}
 
	// add some edges
	for (size_t i = 0; i < N_VERTEX; ++i)
	{
		for (size_t j = 0; j < N_VERTEX; ++j)
		{
			if (i == j) continue;
			if (real_poses[i].distanceTo(real_poses[j]) < DIST_THRES)
				local::add_edge(i, j, real_poses, graph_links, cov);
		}
	}
 
	// Dijkstra
	tictac.Tic();
	const size_t SOURCE_NODE = 13;
 
	mrpt::math::CDijkstra<mrpt::poses::CPosePDFGaussian> aDijkstra(
		graph_links,
		SOURCE_NODE,
		local::get_Dijkstra_weight
	);
 
	std::cout << "Dijkstra took " << tictac.Tac()*1e3 << " ms for " << graph_links.edges.size() << " edges." << std::endl;
 
	// display results graphically:
	mrpt::gui::CDisplayWindowPlots win("Dijkstra example");
 
	win.hold_on();
	win.axis_equal();
 
	for (size_t i = 0; i < N_VERTEX; ++i)
	{
		if (SOURCE_NODE == i) continue;
 
		mrpt::math::CDijkstra<mrpt::poses::CPosePDFGaussian>::TListEdges path;
		aDijkstra.getShortestPathTo(i, path);
 
		std::cout << "to " << i << " -> #steps= " << path.size() << std::endl;
 
		win.clf();
 
		// plot all edges:
		for (mrpt::poses::CNetworkOfPoses2D::iterator e = graph_links.begin(); e != graph_links.end(); ++e)
		{
			const mrpt::poses::CPose2D &p1 = real_poses[e->first.first];
			const mrpt::poses::CPose2D &p2 = real_poses[e->first.second];
 
			mrpt::vector_float X(2);
			mrpt::vector_float Y(2);
			X[0] = p1.x();  Y[0] = p1.y();
			X[1] = p2.x();  Y[1] = p2.y();
			win.plot(X, Y, "k1");
		}
 
		// draw the shortest path:
		for (mrpt::math::CDijkstra<mrpt::poses::CPosePDFGaussian>::TListEdges::const_iterator a = path.begin(); a != path.end(); ++a)
		{
			const mrpt::poses::CPose2D &p1 = real_poses[a->first];
			const mrpt::poses::CPose2D &p2 = real_poses[a->second];
 
			mrpt::vector_float X(2);
			mrpt::vector_float Y(2);
			X[0] = p1.x();  Y[0] = p1.y();
			X[1] = p2.x();  Y[1] = p2.y();
			win.plot(X, Y, "g3");
		}
 
		// draw All nodes:
		win.plot(xs, ys, ".b7");
		win.axis_fit(true);
	}
 
	std::cout << "press any key to continue...";
	win.waitForKey();
}
