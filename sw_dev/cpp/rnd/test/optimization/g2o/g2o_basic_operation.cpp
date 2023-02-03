//#include "stdafx.h"
#include <iterator>
#include <algorithm>
#include <string>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/factory.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/solvers/structure_only/structure_only_solver.h>


namespace {
namespace local {

#if defined(G2O_HAVE_CHOLMOD)
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#elif defined(G2O_HAVE_CSPARSE)
G2O_USE_OPTIMIZATION_LIBRARY(csparse);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif
G2O_USE_OPTIMIZATION_LIBRARY(dense);
G2O_USE_OPTIMIZATION_LIBRARY(pcg);
G2O_USE_OPTIMIZATION_LIBRARY(structure_only);

//G2O_USE_TYPE_GROUP(icp)
//G2O_USE_TYPE_GROUP(sba)
//G2O_USE_TYPE_GROUP(sim3)
//G2O_USE_TYPE_GROUP(slam2d)
G2O_USE_TYPE_GROUP(slam3d)

void g2o_file_interface_test()
{
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<3, 2> >;  // PoseDim & LandmarkDim.
	//using block_solver_type = g2o::BlockSolver_3_2;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >;
	using block_solver_type = g2o::BlockSolver_6_3;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<7, 3> >;
	//using block_solver_type = g2o::BlockSolver_7_3;
	//using block_solver_type = g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >;
	//using block_solver_type = g2o::BlockSolverX;

	//using linear_solver_type = g2o::LinearSolverDense<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverEigen<block_solver_type::PoseMatrixType>;
	using linear_solver_type = g2o::LinearSolverCholmod<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverCSparse<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::LinearSolverPCG<block_solver_type::PoseMatrixType>;
	//using linear_solver_type = g2o::StructureOnlySolver<3>;  // PointDoF.

	auto linearSolver = g2o::make_unique<linear_solver_type>();
	//linearSolver->setBlockOrdering(false);

	//auto solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<block_solver_type>(std::move(linearSolver)));
	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<block_solver_type>(std::move(linearSolver)));

	g2o::SparseOptimizer optimizer;
	optimizer.setAlgorithm(solver);

	// g2o_simulator2d & g2o_simulator3d can be used to generate .g2o files.
	const std::string input_filename("/path/to/input.g2o");
	if (!optimizer.load(input_filename.c_str()))
	{
		std::cerr << "Failed to load a file, " << input_filename << std::endl;
		return;
	}

	std::cout << "#vertices = " << optimizer.vertices().size() << std::endl;
	std::cout << "#edges = " << optimizer.edges().size() << std::endl;
	std::cout << "chi2 = " << optimizer.chi2() << std::endl;
	std::cout << "Max dimension = " << optimizer.maxDimension() << std::endl;
	/*
	std::cout << "Dimensions = ";
	std::copy(optimizer.dimensions().begin(), optimizer.dimensions().end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;
	*/

	/*
	const std::string output_filename("/path/to/before_optimzation.g2o");
	if (!optimizer.save(output_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_filename << std::endl;
		return;
	}
	*/

	// Optimize.
	const int max_iterations = 1000;
	const bool online = false;
	const bool verbose = true;

	if (!optimizer.initializeOptimization())
	{
		std::cerr << "Optimizer not initialized." << std::endl;
		return;
	}
	optimizer.setVerbose(verbose);

	std::cout << "Optimizing..." << std::endl;
	optimizer.optimize(max_iterations, online);
	std::cout << "Optimized." << std::endl;

	const std::string output_filename("/path/to/after_optimzation.g2o");
	if (!optimizer.save(output_filename.c_str()))
	{
		std::cerr << "Failed to save to " << output_filename << std::endl;
		return;
	}

	// Free the graph memory.
	optimizer.clear();
}

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

void basic_operation()
{
	local::g2o_file_interface_test();
}

}  // namespace my_g2o
