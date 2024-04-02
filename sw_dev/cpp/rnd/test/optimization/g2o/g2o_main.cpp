//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

void basic_operation();

void circle_fit_example();
void curve_fit_example();

void gicp_example();
void gicp_sba_example();

void ba_example();
void sba_example();
void bal_example();

void slam2d_tutorial();

void simple_slam3d_test();
void slam3d_se3_test();
void slam3d_se3_pointxyz_test();

}  // namespace my_g2o

int g2o_main(int argc, char *argv[])
{
	// Linear solver.
	//	A * x = b.
	//
	//	g2o::LinearSolverDense:
	//	g2o::LinearSolverEigen:
	//		A: general.
	//	g2o::LinearSolverCholmod: 
	//		A: symmetric.
	//		Cholesky factorization.
	//	g2o::LinearSolverCSparse: 
	//		A: sparse.

	// Block solver.
	//	g2o::BlockSolver_3_2 (g2o::BlockSolver<g2o::BlockSolverTraits<3, 2> >):
	//		Solver for 2D SLAM.
	//		Pose: 3-DoF (x, y, theta).
	//		Landmark: 2-DoF (x, y).
	//	g2o::BlockSolver_6_3 (g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> >): 
	//		Solver for 3D SLAM & BA.
	//		Pose: 6-DoF (SE3).
	//		Landmark: 3-DoF (x, y, z).
	//	g2o::BlockSolver_7_3 (g2o::BlockSolver<g2o::BlockSolverTraits<7, 3> >): 
	//		Solver for BA with scale.
	//		Pose: 7-DoF (Sim3: SE3 + scale).
	//		Landmark: 3-DoF (x, y, z).
	//		Refer to "Scale Drift-Aware Large Scale Monocular SLAM" RSS 2010.
	//	g2o::BlockSolverX (g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >):
	//		Variable-size solver.

	// g2o::OptimizableGraph::Vertex::setMarginalize().
	//	The setMarginalize() option allows you to make use of the Schur Complement trick to speedup the Bundle Adjustment optimization.
	//	By separating the camera poses and the landmarks in your optimization problem you can take advantage of the problem's matrix structure.
	//	Usually you have many more landmarks than camera poses in Bundle Adjustment problems so, by setting every landmarks as setMarginalize(true), g2o will first estimate the camera poses in a forward pass and then use the resulting poses to optimize the landmarks in a backward pass.
	//	The Schur Complement uses the problem sparsity to speedup its computation and implicitly computes the approximated Hessian to solve the optimization problem.
	//	It is this approximated Hessian that is given and computed by computeMarginals().

	//my_g2o::basic_operation();

	//-----
	// Data fitting.

	//my_g2o::circle_fit_example();
	//my_g2o::curve_fit_example();

	//-----
	// Generalized ICP (GICP).

	//my_g2o::gicp_example();
	//my_g2o::gicp_sba_example();  // Sparse bundle adjustment (SBA).

	//-----
	// Bundle adjustment (BA).
	//	The problem of jointly solving the 3D structures (i.e., location of landmarks or feature points) and camera poses.

	//my_g2o::ba_example();
	//my_g2o::sba_example();  // Sparse bundle adjustment (SBA).
	//my_g2o::bal_example();  // Bundle Adjustment in the Large (BAL).

	//-----
	// Pose graph optimization (PGO).
	//	The problem of estimating a set of camera poses from pairwise relative measurements.

	// Refer to my_g2o::slam2d_tutorial(), my_g2o::slam3d_se3_test(), or my_g2o::slam3d_se3_pointxyz_test().

	// REF [site] >> https://github.com/uoip/g2opy
	//my_g2o::pgo_test();  // Not yet implemented.

	//-----
	// 2D SLAM.

	//my_g2o::slam2d_tutorial();

	// REF [site] >> https://github.com/RainerKuemmerle/g2o/tree/master/g2o/examples/slam2d
	//my_g2o::slam2d_example();  // Not yet implemented.

	//-----
	// 3D SLAM.

	//my_g2o::simple_slam3d_test();
	my_g2o::slam3d_se3_test();
	//my_g2o::slam3d_se3_pointxyz_test();

	return 0;
}
