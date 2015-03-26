//#include "stdafx.h"
#if !defined(__FUNCTION__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __FUNCTION__ L""
//#else
#define __FUNCTION__ ""
//#endif
#endif
#if !defined(__func__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __func__ L""
//#else
#define __func__ ""
//#endif
#endif

#include <scythestat/rng/mersenne.h>
#include <scythestat/distributions.h>
#include <scythestat/ide.h>
#include <scythestat/la.h>
#include <scythestat/matrix.h>
#include <scythestat/rng.h>
#include <scythestat/smath.h>
#include <scythestat/stat.h>
#include <scythestat/optimize.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

class PoissonModel
{
public:
	double operator()(const scythe::Matrix<double> &beta)
	{
		const int n = y_.rows();
		scythe::Matrix<double> eta = X_ * beta;
		scythe::Matrix<double> m = scythe::exp(eta);
		double loglike = 0.0;
		for (int i = 0; i < n; ++i)
			loglike += y_(i) * std::log(m(i)) - m(i);
		return -1.0 * loglike;
	}

	scythe::Matrix<double> y_;
	scythe::Matrix<double> X_;
};

}  // namespace local
}  // unnamed namespace

namespace my_scythe {

// [ref] "The Scythe Statistical Library: An Open Source C++ Library for Statistical Computation", Daniel Pemstein, Kevin M. Quinn, and Andrew D. Martin, JSS 2011.
void parametric_bootstrap_example()
{
	scythe::mersenne myrng;
	const int n = 5;

	scythe::Matrix<double> y = scythe::seqa(5, 1, n);
	scythe::Matrix<double> X(n, 3, false);
	X = 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 0, 1, 0, 1, 0;

	local::PoissonModel poisson_model;
	poisson_model.y_ = y;
	poisson_model.X_ = X;

	scythe::Matrix<double> theta = scythe::invpd(scythe::crossprod(X)) * scythe::t(X) * scythe::log(y);
	scythe::Matrix<double> beta_MLE = scythe::BFGS(poisson_model, theta, myrng, 100, 1e-5, true);

	const int M = 10000;
	scythe::Matrix<double> beta_bs_store(M, 3);
	std::cout << "start processing ..." << std::endl;
	for (int i = 0; i < M; ++i)
	{
		scythe::Matrix<double> eta = X * beta_MLE;
		scythe::Matrix<double> m = scythe::exp(eta);
		for (int j = 0; j < n; ++j)
			poisson_model.y_(j) = myrng.rpois(m(j));
		beta_bs_store(i, scythe::_) = scythe::BFGS(poisson_model, beta_MLE, myrng, 100, 1e-5);
	}
	std::cout << "end processing ..." << std::endl;

	std::cout << "The MLEs are: " << std::endl;
	std::cout << scythe::t(beta_MLE) << std::endl;
	std::cout << "The bootstrap SEs are: " << std::endl;
	std::cout << scythe::sdc(beta_bs_store) << std::endl;

	beta_bs_store.save("./data/statistics/scythe/bootstrap_out.txt");
}

}  // namespace my_scythe
