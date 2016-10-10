//#include "stdafx.h"
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <iostream>
#include <cmath>
#include <cassert>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

// REF [site] >> https://www.gnu.org/software/gsl/manual/html_node/Example-programs-for-B_002dsplines.html#Example-programs-for-B_002dsplines
void spline()
{
	// Number of data points to fit.
	const size_t N = 200;

	// Number of fit coefficients.
	const size_t NCOEFFS = 12;

	// nbreak = ncoeffs + 2 - k = ncoeffs - 2 since k = 4.
	const size_t NBREAK = NCOEFFS - 2;

	const size_t n = N;
	const size_t ncoeffs = NCOEFFS;
	const size_t nbreak = NBREAK;

	gsl_rng_env_setup();
	gsl_rng *r = gsl_rng_alloc(gsl_rng_default);

	// Allocate a cubic bspline workspace (k = 4).
	gsl_bspline_workspace *bw = gsl_bspline_alloc(4, nbreak);
	gsl_vector *B = gsl_vector_alloc(ncoeffs);

	gsl_vector *x = gsl_vector_alloc(n);
	gsl_vector *y = gsl_vector_alloc(n);
	gsl_matrix *X = gsl_matrix_alloc(n, ncoeffs);
	gsl_vector *c = gsl_vector_alloc(ncoeffs);
	gsl_vector *w = gsl_vector_alloc(n);
	gsl_matrix *cov = gsl_matrix_alloc(ncoeffs, ncoeffs);
	gsl_multifit_linear_workspace *mw = gsl_multifit_linear_alloc(n, ncoeffs);

	std::cout << "#m=0,S=0" << std::endl;
	// The data to be fitted.
	for (int i = 0; i < n; ++i)
	{
		const double xi = (15.0 / (N - 1)) * i;
		double yi = std::cos(xi) * std::exp(-0.1 * xi);

		const double sigma = 0.1 * yi;
		const double dy = gsl_ran_gaussian(r, sigma);
		yi += dy;

		gsl_vector_set(x, i, xi);
		gsl_vector_set(y, i, yi);
		gsl_vector_set(w, i, 1.0 / (sigma * sigma));

		std::cout << xi << ", " << yi << std::endl;
	}

	// Use uniform breakpoints on [0, 15].
	gsl_bspline_knots_uniform(0.0, 15.0, bw);

	// Construct the fit matrix X.
	for (int i = 0; i < n; ++i)
	{
		const double xi = gsl_vector_get(x, i);

		// Compute B_j(xi) for all j.
		gsl_bspline_eval(xi, B, bw);

		// Fill in row i of X.
		for (int j = 0; j < ncoeffs; ++j)
		{
			const double Bj = gsl_vector_get(B, j);
			gsl_matrix_set(X, i, j, Bj);
		}
	}

	// Do the fit.
	double chisq;
	gsl_multifit_wlinear(X, w, y, c, cov, &chisq, mw);

	const double dof = n - ncoeffs;
	const double tss = gsl_stats_wtss(w->data, 1, y->data, 1, y->size);
	const double Rsq = 1.0 - chisq / tss;

	std::cerr << "chisq/dof = " << (chisq / dof) << ", Rsq = " << Rsq << std::endl;

	// Output the smoothed curve.
	{
		double yi, yerr;

		std::cout << "#m=1,S=0" << std::endl;
		for (double xi = 0.0; xi < 15.0; xi += 0.1)
		{
			gsl_bspline_eval(xi, B, bw);
			gsl_multifit_linear_est(B, c, cov, &yi, &yerr);
			std::cout << xi << ", " << yi << std::endl;
		}
	}

	gsl_rng_free(r);
	gsl_bspline_free(bw);
	gsl_vector_free(B);
	gsl_vector_free(x);
	gsl_vector_free(y);
	gsl_matrix_free(X);
	gsl_vector_free(c);
	gsl_vector_free(w);
	gsl_matrix_free(cov);
	gsl_multifit_linear_free(mw);
}

}  // namespace my_gsl
