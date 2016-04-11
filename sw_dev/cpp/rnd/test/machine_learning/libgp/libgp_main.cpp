//include "stdafx.h"
#include <libgp/gp.h>
#include <libgp/gp_utils.h>
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${LIBGP_HOME}/examples/gp_example_dense.cc
void dense_example()
{
	// initialize Gaussian process for 2-D input using the squared exponential covariance function with additive white noise.
	libgp::GaussianProcess gp(2, "CovSum(CovSEiso, CovNoise)");

	// initialize hyper parameter vector.
	Eigen::VectorXd params(gp.covf().get_param_dim());
	params << 0.0, 0.0, -2.0;
	// set parameters of covariance function.
	gp.covf().set_loghyper(params);

	// add training patterns.
	const int n = 4000;
	double y;
	for (int i = 0; i < n; ++i)
	{
		const double x[] = { libgp::Utils::drand48() * 4 - 2, libgp::Utils::drand48() * 4 - 2 };
		y = libgp::Utils::hill(x[0], x[1]) + libgp::Utils::randn() * 0.1;
		gp.add_pattern(x, y);
	}

	// total squared error.
	const int m = 1000;
	double tss = 0.0, error, f;
	for (int i = 0; i < m; ++i)
	{
		const double x[] = { libgp::Utils::drand48() * 4 - 2, libgp::Utils::drand48() * 4 - 2 };
		f = gp.f(x);
		y = libgp::Utils::hill(x[0], x[1]);
		error = f - y;
		tss += error * error;
	}
	std::cout << "mse = " << tss / m << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_libgp {

}  // namespace my_libgp

int libgp_main(int argc, char *argv[])
{
	local::dense_example();

	return 0;
}
