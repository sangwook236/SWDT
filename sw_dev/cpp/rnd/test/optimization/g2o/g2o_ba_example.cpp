//#include "stdafx.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <Eigen/Core>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>
#include <g2o/stuff/sampler.h>


namespace {
namespace local {

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

class Sample
{
public:
	static int uniform(int from, int to)
	{
		return static_cast<int>(g2o::Sampler::uniformRand(from, to));
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp
void ba_example()
{
	const double PIXEL_NOISE = 1.0;
	const double OUTLIER_RATIO = 0.0;
	const bool ROBUST_KERNEL = false;
	const bool STRUCTURE_ONLY = false;
	const bool DENSE = false;

	std::cout << "PIXEL_NOISE: " << PIXEL_NOISE << std::endl;
	std::cout << "OUTLIER_RATIO: " << OUTLIER_RATIO << std::endl;
	std::cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << std::endl;
	std::cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY << std::endl;
	std::cout << "DENSE: " << DENSE << std::endl;

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	std::string solverName = "lm_fix6_3";
	if (DENSE)
	{
		solverName = "lm_dense6_3";
	}
	else
	{
#ifdef G2O_HAVE_CHOLMOD
		solverName = "lm_fix6_3_cholmod";
#else
		solverName = "lm_fix6_3";
#endif
	}
	g2o::OptimizationAlgorithmProperty solverProperty;
	optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty));

	std::vector<Eigen::Vector3d> true_points;
	for (size_t i = 0; i < 500; ++i)
	{
		true_points.push_back(Eigen::Vector3d((g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3.0, g2o::Sampler::uniformRand(0.0, 1.0) - 0.5, g2o::Sampler::uniformRand(0.0, 1.0) + 3.0));
	}

	const double focal_length = 1000.0;
	const Eigen::Vector2d principal_point(320.0, 240.0);

	std::vector<g2o::SE3Quat, Eigen::aligned_allocator<g2o::SE3Quat> > true_poses;
	auto cam_params = new g2o::CameraParameters(focal_length, principal_point, 0.0);
	cam_params->setId(0);

	if (!optimizer.addParameter(cam_params))
	{
		assert(false);
	}

	int vertex_id = 0;
	for (size_t i = 0; i < 15; ++i)
	{
		const Eigen::Vector3d trans(i * 0.04 - 1.0, 0, 0);

		Eigen::Quaterniond q;
		q.setIdentity();
		const g2o::SE3Quat pose(q, trans);

		auto v_se3 = new g2o::VertexSE3Expmap();
		v_se3->setId(vertex_id);
		if (i < 2) v_se3->setFixed(true);
		v_se3->setEstimate(pose);
		optimizer.addVertex(v_se3);
		true_poses.push_back(pose);
		++vertex_id;
	}

	int point_id = vertex_id;
	int point_num = 0;
	double sum_diff2 = 0.0;

	std::cout << std::endl;
	std::unordered_map<int, int> pointid_2_trueid;
	std::unordered_set<int> inliers;

	for (size_t i = 0; i < true_points.size(); ++i)
	{
		auto v_p = new g2o::VertexPointXYZ();
		v_p->setId(point_id);
		v_p->setMarginalized(true);
		v_p->setEstimate(true_points.at(i) + Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, 1.0), g2o::Sampler::gaussRand(0.0, 1.0), g2o::Sampler::gaussRand(0.0, 1.0)));

		int num_obs = 0;
		for (size_t j = 0; j < true_poses.size(); ++j)
		{
			const Eigen::Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));
			if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
			{
				++num_obs;
			}
		}

		if (num_obs >= 2)
		{
			optimizer.addVertex(v_p);

			bool inlier = true;
			for (size_t j = 0; j < true_poses.size(); ++j)
			{
				Eigen::Vector2d z = cam_params->cam_map(true_poses.at(j).map(true_points.at(i)));

				if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
				{
					const double sam = g2o::Sampler::uniformRand(0.0, 1.0);
					if (sam < OUTLIER_RATIO)
					{
						z = Eigen::Vector2d(local::Sample::uniform(0, 640), local::Sample::uniform(0, 480));
						inlier = false;
					}
					z += Eigen::Vector2d(g2o::Sampler::gaussRand(0.0, PIXEL_NOISE), g2o::Sampler::gaussRand(0.0, PIXEL_NOISE));

					auto e = new g2o::EdgeProjectXYZ2UV();
					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(j)->second));
					e->setMeasurement(z);
					e->information() = Eigen::Matrix2d::Identity();
					if (ROBUST_KERNEL)
					{
						auto rk = new g2o::RobustKernelHuber;
						e->setRobustKernel(rk);
					}
					e->setParameterId(0, 0);
					optimizer.addEdge(e);
				}
			}

			if (inlier)
			{
				inliers.insert(point_id);
				Eigen::Vector3d diff = v_p->estimate() - true_points[i];

				sum_diff2 += diff.dot(diff);
			}

			pointid_2_trueid.insert(std::make_pair(point_id, i));

			++point_id;
			++point_num;
		}
	}

	std::cout << std::endl;
	optimizer.initializeOptimization();
	optimizer.setVerbose(true);

	if (STRUCTURE_ONLY)
	{
		std::cout << "Performing structure-only BA:" << std::endl;
		g2o::StructureOnlySolver<3> structure_only_ba;
		g2o::OptimizableGraph::VertexContainer points;
		for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it)
		{
			auto v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
			if (v->dimension() == 3) points.push_back(v);
		}
		structure_only_ba.calc(points, 10);
	}
	//optimizer.save("./test.g2o");

	std::cout << std::endl;
	std::cout << "Performing full BA:" << std::endl;
	optimizer.optimize(10);

	std::cout << std::endl;
	std::cout << "Point error before optimisation (inliers only): " << std::sqrt(sum_diff2 / inliers.size()) << std::endl;

	point_num = 0;
	sum_diff2 = 0.0;
	for (std::unordered_map<int, int>::iterator it = pointid_2_trueid.begin(); it != pointid_2_trueid.end(); ++it)
	{
		g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer.vertices().find(it->first);
		if (v_it == optimizer.vertices().end())
		{
			std::cerr << "Vertex " << it->first << " not in graph!" << std::endl;
			exit(-1);
		}
		auto v_p = dynamic_cast<g2o::VertexPointXYZ*>(v_it->second);
		if (v_p == nullptr)
		{
			std::cerr << "Vertex " << it->first << "is not a PointXYZ!" << std::endl;
			exit(-1);
		}

		const Eigen::Vector3d diff = v_p->estimate() - true_points[it->second];
		if (inliers.find(it->first) == inliers.end()) continue;
		sum_diff2 += diff.dot(diff);
		++point_num;
	}

	std::cout << "Point error after optimisation (inliers only): " << std::sqrt(sum_diff2 / inliers.size()) << std::endl;
	std::cout << std::endl;
}

}  // namespace my_g2o
