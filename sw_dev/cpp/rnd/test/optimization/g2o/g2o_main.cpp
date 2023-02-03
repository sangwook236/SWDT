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

	//my_g2o::basic_operation();

	//-----
	// Examples.

	// Data fitting.
	//my_g2o::circle_fit_example();
	//my_g2o::curve_fit_example();

	// GICP.
	//my_g2o::gicp_example();
	//my_g2o::gicp_sba_example();  // Stereo (not sparse) bundle adjustment (SBA).

	// Bundle adjustment (BA).
	//my_g2o::ba_example();
	// Stereo (not sparse) bundle adjustment (SBA).
	//my_g2o::sba_example();
	// Bundle Adjustment in the Large (BAL).
	//my_g2o::bal_example();

	// SLAM 2D.
	// REF [site] >> https://github.com/RainerKuemmerle/g2o/tree/master/g2o/examples/slam2d
	//my_g2o::slam2d_example();  // Not yet implemented.

	// Pose graph optimization (PGO).
	//my_g2o::pgo_test();  // Not yet implemented.

	return 0;
}
