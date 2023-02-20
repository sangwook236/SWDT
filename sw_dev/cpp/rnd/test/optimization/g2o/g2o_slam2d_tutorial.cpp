//#include "stdafx.h"
#include <cmath>
#include <iostream>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam2d/vertex_point_xy.h>
#include <g2o/types/slam2d/vertex_se2.h>
#include <g2o/types/slam2d/edge_se2_pointxy.h>
#include <g2o/types/slam2d/edge_se2.h>
#include <g2o/types/slam2d/parameter_se2_offset.h>
// REF [file] >> https://github.com/RainerKuemmerle/g2o/tree/master/g2o/examples/tutorial_slam2d/simulator.h
#include "simulator.h"


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://github.com/RainerKuemmerle/g2o/tree/master/g2o/examples/tutorial_slam2d
// REF [pdf] >> Ch.8 in g2o.pdf.
void slam2d_tutorial()
{
	// TODO simulate different sensor offset.
	// Simulate a robot observing landmarks while travelling on a grid.
	g2o::SE2 sensorOffsetTransf(0.2, 0.1, -0.1);
	const int numNodes = 300;
	g2o::tutorial::Simulator simulator;
	simulator.simulate(numNodes, sensorOffsetTransf);

	//-----
	// Create the optimization problem.

	typedef g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> > SlamBlockSolver;
	typedef g2o::LinearSolverEigen<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

	// Allocate the optimizer.
	g2o::SparseOptimizer optimizer;
	auto linearSolver = g2o::make_unique<SlamLinearSolver>();
	linearSolver->setBlockOrdering(false);
	auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));

	optimizer.setAlgorithm(solver);

	// Add the parameter representing the sensor offset.
	auto sensorOffset = new g2o::ParameterSE2Offset;
	sensorOffset->setOffset(sensorOffsetTransf);
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);

	// Add the odometry to the optimizer.
	// First: add all the vertices.
	std::cout << "Optimization: add robot poses ... ";
	for (size_t i = 0; i < simulator.poses().size(); ++i)
	{
		const g2o::tutorial::Simulator::GridPose& p = simulator.poses()[i];

		const g2o::SE2& t = p.simulatorPose;
		auto robot = new g2o::VertexSE2;
		robot->setId(p.id);
		robot->setEstimate(t);

		optimizer.addVertex(robot);
	}
	std::cout << "done." << std::endl;

	// Add the landmark vertices.
	std::cout << "Optimization: add landmark vertices ... ";
	for (size_t i = 0; i < simulator.landmarks().size(); ++i)
	{
		const g2o::tutorial::Simulator::Landmark& l = simulator.landmarks()[i];

		auto landmark = new g2o::VertexPointXY;
		landmark->setId(l.id);
		landmark->setEstimate(l.simulatedPose);

		optimizer.addVertex(landmark);
	}
	std::cout << "done." << std::endl;

	// Second: add the odometry constraints.
	std::cout << "Optimization: add odometry measurements ... ";
	for (size_t i = 0; i < simulator.odometry().size(); ++i)
	{
		const g2o::tutorial::Simulator::GridEdge& simEdge = simulator.odometry()[i];

		auto odometry = new g2o::EdgeSE2;
		odometry->vertices()[0] = optimizer.vertex(simEdge.from);
		odometry->vertices()[1] = optimizer.vertex(simEdge.to);
		odometry->setMeasurement(simEdge.simulatorTransf);
		odometry->setInformation(simEdge.information);

		optimizer.addEdge(odometry);
	}
	std::cout << "done." << std::endl;

	// Add landmark constraints.
	std::cout << "Optimization: add landmark observations ... ";
	for (size_t i = 0; i < simulator.landmarkObservations().size(); ++i)
	{
		const g2o::tutorial::Simulator::LandmarkEdge& simEdge = simulator.landmarkObservations()[i];

		auto landmarkObservation = new g2o::EdgeSE2PointXY;
		landmarkObservation->vertices()[0] = optimizer.vertex(simEdge.from);
		landmarkObservation->vertices()[1] = optimizer.vertex(simEdge.to);
		landmarkObservation->setMeasurement(simEdge.simulatorMeas);
		landmarkObservation->setInformation(simEdge.information);
		landmarkObservation->setParameterId(0, sensorOffset->id());

		optimizer.addEdge(landmarkObservation);
	}
	std::cout << "done." << std::endl;

	//-----
	// Optimization.

	// Dump initial state to the disk.
	optimizer.save("../tutorial_slam2d_before.g2o");

	// Prepare and run the optimization.
	// Fix the first robot pose to account for gauge freedom.
	auto firstRobotPose = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(0));
	firstRobotPose->setFixed(true);

	std::cout << "Optimizing" << std::endl;
	optimizer.initializeOptimization();
	optimizer.setVerbose(true);
	optimizer.optimize(10);
	std::cout << "done." << std::endl;

	optimizer.save("../tutorial_slam2d_after.g2o");

	// Free the graph memory.
	optimizer.clear();
}

}  // namespace my_g2o
