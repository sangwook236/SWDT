//#include "stdafx.h"
#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/icp/types_icp.h>
#include <g2o/stuff/sampler.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/icp/gicp_demo.cpp
void gicp_example()
{
	const double euc_noise = 0.01;  // Noise in position, m.
	//const double outlier_ratio = 0.1;

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	// Variable-size block solver.
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>()));

	optimizer.setAlgorithm(solver);

	std::vector<Eigen::Vector3d> true_points;
	for (size_t i = 0; i < 1000; ++i)
	{
		true_points.push_back(Eigen::Vector3d((g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3.0, g2o::Sampler::uniformRand(0.0, 1.0) - 0.5, g2o::Sampler::uniformRand(0.0, 1.0) + 10.0));
	}

	// Set up two poses.
	int vertex_id = 0;
	for (size_t i = 0; i < 2; ++i)
	{
		// Set up rotation and translation for this node.
		Eigen::Vector3d t(0, 0, double(i));
		Eigen::Quaterniond q;
		q.setIdentity();

		Eigen::Isometry3d cam;  // Camera pose.
		cam = q;
		cam.translation() = t;

		// Set up node.
		auto vc = new g2o::VertexSE3();
		vc->setEstimate(cam);
		vc->setId(vertex_id);  // Vertex ID.

		std::cerr << t.transpose() << " | " << q.coeffs().transpose() << std::endl;

		// Set first cam pose fixed.
		if (i == 0) vc->setFixed(true);

		// Add to optimizer.
		optimizer.addVertex(vc);

		++vertex_id;
	}

	// Set up point matches.
	for (size_t i = 0; i < true_points.size(); ++i)
	{
		// Get two poses.
		auto vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
		auto vp1 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);

		// Calculate the relative 3D position of the point.
		Eigen::Vector3d pt0, pt1;
		pt0 = vp0->estimate().inverse() * true_points[i];
		pt1 = vp1->estimate().inverse() * true_points[i];

		// Add in noise.
		pt0 += Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise));
		pt1 += Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise));

		// Form edge, with normals in varioius positions.
		Eigen::Vector3d nm0, nm1;
		nm0 << 0, i, 1;
		nm1 << 0, i, 1;
		nm0.normalize();
		nm1.normalize();

		auto e = new g2o::Edge_V_V_GICP();  // New edge with correct cohort for caching.
		e->setVertex(0, vp0);  // First viewpoint.
		e->setVertex(1, vp1);  // Second viewpoint.

		g2o::EdgeGICP meas;
		meas.pos0 = pt0;
		meas.pos1 = pt1;
		meas.normal0 = nm0;
		meas.normal1 = nm1;

		e->setMeasurement(meas);
		//e->inverseMeasurement().pos() = -kp;
		meas = e->measurement();

		// Use this for point-plane.
		e->information() = meas.prec0(0.01);
		// Use this for point-point.
		//e->information().setIdentity();

		//e->setRobustKernel(true);
		//e->setHuberWidth(0.01);

		optimizer.addEdge(e);
	}

	// Move second cam off of its true position.
	auto vc = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);
	Eigen::Isometry3d cam = vc->estimate();
	cam.translation() = Eigen::Vector3d(0, 0, 0.2);
	vc->setEstimate(cam);

	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	std::cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << std::endl;
	optimizer.setVerbose(true);

	optimizer.optimize(5);

	std::cout << std::endl << "Second vertex should be near 0,0,1" << std::endl;
	std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second)->estimate().translation().transpose() << std::endl;
	std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second)->estimate().translation().transpose() << std::endl;
}

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/icp/gicp_sba_demo.cpp
void gicp_sba_example()
{
	const int num_points = 500;  // # of points to use in projection SBA.
	const double euc_noise = 0.1;  // Noise in position, m.
	const double pix_noise = 1.0;  // Pixel noise.
	//const double outlier_ratio = 0.1;

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	// Variable-size block solver.
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>()));

	optimizer.setAlgorithm(solver);

	std::vector<Eigen::Vector3d> true_points;
	for (size_t i = 0; i < 1000; ++i)
	{
		true_points.push_back(Eigen::Vector3d((g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3.0, g2o::Sampler::uniformRand(0.0, 1.0) - 0.5, g2o::Sampler::uniformRand(0.0, 1.0) + 10.0));
	}

	// Set up camera params.
	const Eigen::Vector2d focal_length(500, 500);  // [pixels].
	const Eigen::Vector2d principal_point(320, 240);  // 640x480 image.
	const double baseline = 0.075;  // 7.5 cm baseline.

	// Set up camera params and projection matrices on vertices.
	g2o::VertexSCam::setKcam(focal_length[0], focal_length[1], principal_point[0], principal_point[1], baseline);

	// Set up two poses.
	int vertex_id = 0;
	for (size_t i = 0; i < 2; ++i)
	{
		// Set up rotation and translation for this node.
		Eigen::Vector3d t(0, 0, double(i));
		Eigen::Quaterniond q;
		q.setIdentity();

		Eigen::Isometry3d cam;  // Camera pose.
		cam = q;
		cam.translation() = t;

		// Set up node.
		auto vc = new g2o::VertexSCam();  // Stereo camera vertex.
		vc->setEstimate(cam);
		vc->setId(vertex_id);  // Vertex ID.

		std::cerr << t.transpose() << " | " << q.coeffs().transpose() << std::endl;

		// Set first cam pose fixed.
		if (i == 0) vc->setFixed(true);

		// Make sure projection matrices are set.
		vc->setAll();

		// Add to optimizer.
		optimizer.addVertex(vc);

		++vertex_id;
	}

	// Set up point matches for GICP.
	for (size_t i = 0; i < true_points.size(); ++i)
	{
		// Get two poses.
		auto vp0 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second);
		auto vp1 = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);

		// Calculate the relative 3D position of the point.
		Eigen::Vector3d pt0, pt1;
		pt0 = vp0->estimate().inverse() * true_points[i];
		pt1 = vp1->estimate().inverse() * true_points[i];

		// Add in noise.
		pt0 += Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise));
		pt1 += Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise), g2o::Sampler::gaussRand(0.0, euc_noise));

		// Form edge, with normals in varioius positions.
		Eigen::Vector3d nm0, nm1;
		nm0 << 0, i, 1;
		nm1 << 0, i, 1;
		nm0.normalize();
		nm1.normalize();

		auto e = new g2o::Edge_V_V_GICP();  // New edge with correct cohort for caching.
		e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(vp0);  // First viewpoint.
		e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(vp1);  // Second viewpoint.

		g2o::EdgeGICP meas;
		meas.pos0 = pt0;
		meas.pos1 = pt1;
		meas.normal0 = nm0;
		meas.normal1 = nm1;

		e->setMeasurement(meas);
		//e->inverseMeasurement().pos() = -kp;
		meas = e->measurement();

		// Use this for point-plane.
		e->information() = meas.prec0(0.01);
		// Use this for point-point.
		//e->information().setIdentity();

		//e->setRobustKernel(true);
		//e->setHuberWidth(0.01);

		optimizer.addEdge(e);
	}

	// Set up SBA projections with some number of points.
	true_points.clear();
	for (int i = 0; i < num_points; ++i)
	{
		true_points.push_back(Eigen::Vector3d((g2o::Sampler::uniformRand(0.0, 1.0) - 0.5) * 3.0, g2o::Sampler::uniformRand(0.0, 1.0) - 0.5, g2o::Sampler::uniformRand(0.0, 1.0) + 10.0));
	}

	// Add point projections to this vertex.
	for (size_t i = 0; i < true_points.size(); ++i)
	{
		auto v_p = new g2o::VertexPointXYZ();
		v_p->setId(vertex_id++);
		v_p->setMarginalized(true);
		v_p->setEstimate(true_points.at(i) + Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, 1.0), g2o::Sampler::gaussRand(0.0, 0.0), g2o::Sampler::gaussRand(0.0, 1.0)));

		optimizer.addVertex(v_p);

		for (size_t j = 0; j < 2; ++j)
		{
			Eigen::Vector3d z;
			dynamic_cast<g2o::VertexSCam*>(optimizer.vertices().find(j)->second)->mapPoint(z, true_points.at(i));  // Calculate stereo projection.

			if (z[0] >= 0 && z[1] >= 0 && z[0] < 640 && z[1] < 480)
			{
				z += Eigen::Vector3d(g2o::Sampler::gaussRand(0.0, pix_noise), g2o::Sampler::gaussRand(0.0, pix_noise), g2o::Sampler::gaussRand(0.0, pix_noise / 16.0));

				auto e = new g2o::Edge_XYZ_VSC();  // Stereo projection.
				e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_p);
				e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertices().find(j)->second);

				e->setMeasurement(z);
				//e->inverseMeasurement() = -z;
				e->information() = Eigen::Matrix3d::Identity();

				//e->setRobustKernel(false);
				//e->setHuberWidth(1);

				optimizer.addEdge(e);
			}
		}
	}  // Done with adding projection points.

	// Move second cam off of its true position.
	auto vc = dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second);
	Eigen::Isometry3d cam = vc->estimate();
	cam.translation() = Eigen::Vector3d(-0.1, 0.1, 0.2);
	vc->setEstimate(cam);

	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	std::cout << "Initial chi2 = " << FIXED(optimizer.chi2()) << std::endl;
	optimizer.setVerbose(true);

	optimizer.optimize(20);

	std::cout << std::endl << "Second vertex should be near 0,0,1" << std::endl;
	std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(0)->second)->estimate().translation().transpose() << std::endl;
	std::cout << dynamic_cast<g2o::VertexSE3*>(optimizer.vertices().find(1)->second)->estimate().translation().transpose() << std::endl;
}

}  // namespace my_g2o
