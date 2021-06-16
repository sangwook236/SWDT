//#include <gsl/gsl_spline2d.h>
//#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_bspline.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_math.h>
#include <iostream>
#include <array>
#include <cmath>
#include <stdexcept>
#include <cassert>


namespace {
namespace local {

// REF [site] >> https://www.gnu.org/software/gsl/manual/html_node/1D-Interpolation-Example-programs.html
void spline_1d_example()
{
	// Compute a (cubic) spline interpolation of the 10-point dataset(x_i, y_i) where x_i = i + sin(i) / 2 and y_i = i + cos(i ^ 2) for i = 0 ... 9.

	//std::cout << "Input data:" << std::endl;
	double x[10], y[10];
	for (int i = 0; i < 10; ++i)
	{
		x[i] = i + 0.5 * std::sin(i);  // x values must be strictly increasing.
		y[i] = i + std::cos(i * i);

		//std::cout << '\t' << x[i] << ',' << y[i] << std::endl;
	}

	//
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, 10);
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_polynomial, 10);
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, 10);  // Cubic spline with natural boundary condition.
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_akima, 10);  // Non-rounded Akima spline with natural boundary condition.

	gsl_spline_init(spline, x, y, 10);

	std::cout << "Spline curve:" << std::endl;
	for (double xi = x[0]; xi < x[9]; xi += 0.01)
	{
		const double yi = gsl_spline_eval(spline, xi, acc);

		std::cout << '\t' << xi << ' ' << yi << std::endl;
	}

	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
}

// REF [site] >> https://www.gnu.org/software/gsl/manual/html_node/1D-Interpolation-Example-programs.html
void spline_1d_periodic_example()
{
	const int N = 4;

	// Note: y[0] == y[3] for periodic data.
	const double x[4] = { 0.00, 0.10,  0.27,  0.30 };  // x values must be strictly increasing.
	const double y[4] = { 0.15, 0.70, -0.10,  0.15 };

	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline_periodic, N);  // Cubic spline with periodic boundary condition.
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_akima_periodic, N);  // Non-rounded Akima spline with periodic boundary condition.

	//std::cout << "Input data:" << std::endl;
	//for (int i = 0; i < N; ++i)
	//	std::cout << '\t' << x[i] << ',' << y[i] << std::endl;

	//
	gsl_spline_init(spline, x, y, N);

	// The slope of the fitted curve is the same at the beginning and end of the data, and the second derivative is also.
	std::cout << "Spline curve:" << std::endl;
	for (int i = 0; i <= 100; ++i)
	{
		const double xi = (1 - i / 100.0) * x[0] + (i / 100.0) * x[N - 1];
		const double yi = gsl_spline_eval(spline, xi, acc);

		std::cout << '\t' << xi << ',' << yi << std::endl;
	}

	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
}

// REF [site] >> https://www.gnu.org/software/gsl/manual/html_node/1D-Interpolation-Example-programs.html
void spline_1d_comparision_example()
{
	const size_t N = 9;

	// This dataset is taken from J. M. Hyman, Accurate Monotonicity preserving cubic interpolation, SIAM J. Sci. Stat. Comput. 4, 4, 1983.
	const double x[] = { 7.99, 8.09, 8.19, 8.7, 9.2, 10.0, 12.0, 15.0, 20.0 };  // x values must be strictly increasing.
	const double y[] = { 0.0, 2.76429e-5, 4.37498e-2, 0.169183, 0.469428, 0.943740, 0.998636, 0.999919, 0.999994 };

	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline *spline_cubic = gsl_spline_alloc(gsl_interp_cspline, N);
	gsl_spline *spline_akima = gsl_spline_alloc(gsl_interp_akima, N);
	//gsl_spline *spline_steffen = gsl_spline_alloc(gsl_interp_steffen, N);

	gsl_spline_init(spline_cubic, x, y, N);
	gsl_spline_init(spline_akima, x, y, N);
	//gsl_spline_init(spline_steffen, x, y, N);

	//std::cout << "Input data:" << std::endl;
	//for (size_t i = 0; i < N; ++i)
	//	std::cout << '\t' << x[i] << ',' << y[i] << std::endl;

	// The cubic method exhibits a local maxima between the 6th and 7th data points and continues oscillating for the rest of the data.
	// Akima also shows a local maxima but recovers and follows the data well after the 7th grid point.
	// Steffen preserves monotonicity in all intervals and does not exhibit oscillations, at the expense of having a discontinuous second derivative.
	std::cout << "Spline curves:" << std::endl;
	for (size_t i = 0; i <= 100; ++i)
	{
		const double xi = (1 - i / 100.0) * x[0] + (i / 100.0) * x[N - 1];
		const double yi_cubic = gsl_spline_eval(spline_cubic, xi, acc);
		const double yi_akima = gsl_spline_eval(spline_akima, xi, acc);
		//const double yi_steffen = gsl_spline_eval(spline_steffen, xi, acc);

		//std::cout << '\t' << xi << ", " << yi_cubic << ", " << yi_akima << ", " << yi_steffen << std::endl;
		std::cout << '\t' << xi << ", " << yi_cubic << ", " << yi_akima << std::endl;
	}

	gsl_spline_free(spline_cubic);
	gsl_spline_free(spline_akima);
	//gsl_spline_free(spline_steffen);
	gsl_interp_accel_free(acc);
}

void spline_2d_example()
{
#if 0
	// Perform bilinear interpolation on the unit square, using z values of (0,1, 0.5, 1) going clockwise around the square.

	const size_t N = 100;  // Number of points to interpolate.

	// Define unit square.
	const double xa[] = { 0.0, 1.0 };  // x values must be strictly increasing.
	const double ya[] = { 0.0, 1.0 };
	const size_t nx = sizeof(xa) / sizeof(double);  // x grid points.
	const size_t ny = sizeof(ya) / sizeof(double);  // y grid points.

	const gsl_interp2d_type *T = gsl_interp2d_bilinear;
	gsl_spline2d *spline = gsl_spline2d_alloc(T, nx, ny);

	gsl_interp_accel *xacc = gsl_interp_accel_alloc();
	gsl_interp_accel *yacc = gsl_interp_accel_alloc();

	// Set z grid values.
	std::array<double, nx * ny> za;
	gsl_spline2d_set(spline, za.data(), 0, 0, 0.0);
	gsl_spline2d_set(spline, za.data(), 0, 1, 1.0);
	gsl_spline2d_set(spline, za.data(), 1, 1, 0.5);
	gsl_spline2d_set(spline, za.data(), 1, 0, 1.0);

	// Initialize interpolation.
	gsl_spline2d_init(spline, xa, ya, za.data(), nx, ny);

	// Interpolate N values in x and y and print out grid for plotting.
	std::cout << "Spline curve:" << std::endl;
	for (size_t i = 0; i < N; ++i)
	{
		const double xi = (double)i / (N - 1.0);

		for (size_t j = 0; j < N; ++j)
		{
			double yj = j / (N - 1.0);
			double zij = gsl_spline2d_eval(spline, xi, yj, xacc, yacc);

			std::cout << '\t' << xi << ',' << yj << ',' << zij << std::endl;
		}
		std::cout << std::endl;
	}

	gsl_spline2d_free(spline);
	gsl_interp_accel_free(xacc);
	gsl_interp_accel_free(yacc);
#else
	throw std::runtime_error("Not yet supported");
#endif
}

// REF [site] >> https://www.gnu.org/software/gsl/manual/html_node/Example-programs-for-B_002dsplines.html
void bspline()
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

	// The data to be fitted.
	//std::cout << "Input data:" << std::endl;
	for (int i = 0; i < n; ++i)
	{
		const double xi = (15.0 / (N - 1)) * i;  // x values must be strictly increasing.
		double yi = std::cos(xi) * std::exp(-0.1 * xi);

		const double sigma = 0.1 * yi;
		const double dy = gsl_ran_gaussian(r, sigma);
		yi += dy;

		gsl_vector_set(x, i, xi);
		gsl_vector_set(y, i, yi);
		gsl_vector_set(w, i, 1.0 / (sigma * sigma));

		//std::cout << '\t' << xi << ", " << yi << std::endl;
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
		std::cout << "Spline curve:" << std::endl;
		double yi, yerr;
		for (double xi = 0.0; xi < 15.0; xi += 0.1)
		{
			gsl_bspline_eval(xi, B, bw);
			gsl_multifit_linear_est(B, c, cov, &yi, &yerr);

			std::cout << '\t' << xi << ", " << yi << std::endl;
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

void spline_1d_differentiation()
{
	const size_t N = 5;
	const double x[] = { -1.0, -0.5, 0.0, +0.5, +1.0 };  // x values must be strictly increasing.
	const double y[] = { +1.0, 0.25, 0.0, 0.25, +1.0 };  // Function values.
	//for (int i = 0; i < N; ++i)
	//	std::cout << x[i] << ',' << y[i] << std::endl;

	//
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, N);
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_polynomial, N);
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, N);  // Cubic spline with natural boundary condition.
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_akima, N);  // Non-rounded Akima spline with natural boundary condition.

	gsl_spline_init(spline, x, y, N);

	//
	const double xi = 0.25;
	const double yi = gsl_spline_eval(spline, xi, acc);
#if 1
	double dyi = 0.0;
	const int retval1 = gsl_spline_eval_deriv_e(spline, xi, acc, &dyi);
	assert(0 == retval1);
	double d2yi = 0.0;
	const int retval2 = gsl_spline_eval_deriv2_e(spline, xi, acc, &d2yi);
	assert(0 == retval2);
#else
	const double dyi = gsl_spline_eval_deriv(spline, xi, acc);
	const double d2yi = gsl_spline_eval_deriv2(spline, xi, acc);
#endif

	std::cout << "Derivative of spline:" << std::endl;
	std::cout << "\tx = " << xi << ", y = " << yi << ", dy = " << dyi << ", d2y = " << d2yi << std::endl;

	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
}

void spline_1d_integration()
{
	const size_t N = 5;
	const double x[] = { -1.0, -0.5, 0.0, +0.5, +1.0 };  // x values must be strictly increasing.
	const double y[] = { +1.0, 0.25, 0.0, 0.25, +1.0 };  // Function values.
	//for (int i = 0; i < N; ++i)
	//	std::cout << x[i] << ',' << y[i] << std::endl;

	//
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, N);
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_polynomial, N);
	//gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, N);  // Cubic spline with natural boundary condition.
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_akima, N);  // Non-rounded Akima spline with natural boundary condition.

	gsl_spline_init(spline, x, y, N);

	//
	double xi = -1.0, xf = 0.25;
#if 1
	double ydx = 0.0;
	const int retval = gsl_spline_eval_integ_e(spline, xi, xf, acc, &ydx);
	assert(0 == retval);
#else
	const double ydx = gsl_spline_eval_integ(spline, xi, xf, acc);
#endif

	std::cout << "Integration of spline:" << std::endl;
	std::cout << "\tx = (" << xi << ',' << xf << "), ydx = " << ydx << std::endl;

	gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
}

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void spline()
{
	// 1-dim spline.
	//local::spline_1d_example();
	//local::spline_1d_periodic_example();
	//local::spline_1d_comparision_example();

	// 2-dim spline.
	//local::spline_2d_example();  // NOTICE [error] >> Not yet supported. (?)

	// B-spline.
	//local::bspline();

	// Differentiation and integration.
	local::spline_1d_differentiation();
	local::spline_1d_integration();
}

}  // namespace my_gsl
