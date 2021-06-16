//#include "stdafx.h"
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_linalg.h>
#include <vector>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void unit_gaussian_distribution()
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

void gaussian_distribution()
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

void bivariate_gaussian_distribution()
{
	{
		const double mu_x = -1.0, mu_y = 2.0, sigma_x = 2.0, sigma_y = std::sqrt(3.0), sigma_xy = -2.25, rho = sigma_xy / (sigma_x * sigma_y);
		const double x = 0.5, y = 1.5;
		std::cout << "p([0.5, 1.5] | [-1.0, 2.0], [4.0, -2.25 ; -2.25, 3.0]) = " << gsl_ran_bivariate_gaussian_pdf(x - mu_x, y - mu_y, sigma_x, sigma_y, rho) << std::endl;
	}

	{
		const double mu_x = 1.0, mu_y = -1.0, sigma_x = std::sqrt(0.9), sigma_y = std::sqrt(0.3), sigma_xy = 0.4, rho = sigma_xy / (sigma_x * sigma_y);
		const double x = 0.5, y = -0.5;
		std::cout << "p([0.5, -0.5] | [1.0, -1.0], [0.9, 0.4 ; 0.4, 0.3]) = " << gsl_ran_bivariate_gaussian_pdf(x - mu_x, y - mu_y, sigma_x, sigma_y, rho) << std::endl;
	}
}

void multivariate_gaussian_distribution()
{
	{
		gsl_vector *mu = gsl_vector_alloc(2);
		gsl_matrix *covL = gsl_matrix_alloc(2, 2);
		gsl_vector *x = gsl_vector_alloc(2);
		gsl_vector *work = gsl_vector_alloc(2);

		gsl_vector_set(mu, 0, -1.0);  gsl_vector_set(mu, 1, 2.0);
		gsl_matrix_set(covL, 0, 0, 4.0);  gsl_matrix_set(covL, 0, 1, -2.25);  gsl_matrix_set(covL, 1, 0, -2.25);  gsl_matrix_set(covL, 1, 1, 3.0);
		gsl_vector_set(x, 0, 0.5);  gsl_vector_set(x, 1, 1.5);

		gsl_linalg_cholesky_decomp(covL);

		double pdf = 0.0;
		gsl_ran_multivariate_gaussian_pdf(x, mu, covL, &pdf, work);
		std::cout << "p([0.5, 1.5] | [-1.0, 2.0], [4.0, -2.25 ; -2.25, 3.0]) = " << pdf << std::endl;

		gsl_vector_free(mu);
		gsl_matrix_free(covL);
		gsl_vector_free(x);
		gsl_vector_free(work);
	}

	{
		gsl_vector *mu = gsl_vector_alloc(2);
		gsl_matrix *covL = gsl_matrix_alloc(2, 2);
		gsl_vector *x = gsl_vector_alloc(2);
		gsl_vector *work = gsl_vector_alloc(2);

		gsl_vector_set(mu, 0, 1.0);  gsl_vector_set(mu, 1, -1.0);
		gsl_matrix_set(covL, 0, 0, 0.9);  gsl_matrix_set(covL, 0, 1, 0.4);  gsl_matrix_set(covL, 1, 0, 0.4);  gsl_matrix_set(covL, 1, 1, 0.3);
		gsl_vector_set(x, 0, 0.5);  gsl_vector_set(x, 1, -0.5);

		gsl_linalg_cholesky_decomp(covL);

		double pdf = 0.0;
		gsl_ran_multivariate_gaussian_pdf(x, mu, covL, &pdf, work);
		std::cout << "p([0.5, -0.5] | [1.0, -1.0], [0.9, 0.4 ; 0.4, 0.3]) = " << pdf << std::endl;

		gsl_vector_free(mu);
		gsl_matrix_free(covL);
		gsl_vector_free(x);
		gsl_vector_free(work);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void distribution()
{
	local::unit_gaussian_distribution();
	local::gaussian_distribution();
	local::bivariate_gaussian_distribution();
	local::multivariate_gaussian_distribution();
}

}  // namespace my_gsl
