//#include "stdafx.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <vector>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void distribution_unit_gaussian()
{
	//
	{
		const int N = 1000;
		std::vector<double> data;
		data.reserve(N);
		for (int i = 0; i < N; ++i)
			data.push_back(gsl_ran_ugaussian_pdf(-3.0 + 6.0 * i / N));
	}

	//
	{
		const double x = 2.0;

		const double P = gsl_cdf_ugaussian_P(x);
		std::cout << "prob(x < " << x << ") = " << P << std::endl;

		const double Q = gsl_cdf_ugaussian_Q(x);
		std::cout << "prob(x < " << x << ") = " << Q << std::endl;

		const double xp = gsl_cdf_ugaussian_Pinv(P);
		std::cout << "Pinv(" << P << ") = " << xp << std::endl;

		const double xq = gsl_cdf_ugaussian_Qinv(Q);
		std::cout << "Qinv(" << Q << ") = " << xq << std::endl;
	}
}

void distribution_gaussian()
{
	//
	{
		const double sigma = 2.0;

		const int N = 1000;
		std::vector<double> data;
		data.reserve(N);
		for (int i = 0; i < N; ++i)
			data.push_back(gsl_ran_gaussian_pdf(-3.0 + 6.0 * i / N, sigma));
	}

	//
	{
		const double sigma = 2.0;
		const double x = 2.0;

		const double P = gsl_cdf_gaussian_P(x, sigma);
		std::cout << "prob(x < " << x << ") = " << P << std::endl;

		const double Q = gsl_cdf_gaussian_Q(x, sigma);
		std::cout << "prob(x < " << x << ") = " << Q << std::endl;

		const double xp = gsl_cdf_gaussian_Pinv(P, sigma);
		std::cout << "Pinv(" << P << ") = " << xp << std::endl;

		const double xq = gsl_cdf_gaussian_Qinv(Q, sigma);
		std::cout << "Qinv(" << Q << ") = " << xq << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace gsl {

void distribution()
{
	local::distribution_unit_gaussian();
	local::distribution_gaussian();
}

}  // namespace gsl
