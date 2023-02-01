//#include "stdafx.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/stuff/command_args.h>
#include <g2o/stuff/sampler.h>
#include <g2o/core/robust_kernel_impl.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

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
	// Data fitting.
	//my_g2o::circle_fit_example();
	//my_g2o::curve_fit_example();

	// GICP.
	//my_g2o::gicp_example();
	//my_g2o::gicp_sba_example();

	// Bundle adjustment (BA).
	my_g2o::ba_example();
	// Sparse bundle adjustment (SBA).
	//my_g2o::sba_example();
	// Bundle Adjustment in the Large (BAL).
	//my_g2o::bal_example();

	// Pose graph optimization (PGO).
	//local::pgo_example();  // Not yet implemented.

	return 0;
}
