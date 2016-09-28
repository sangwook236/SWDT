//#define __USE_GSL_FOR_CHECKING 1
//#include "stdafx.h"
#include <nlopt.hpp>
#if defined(__USE_GSL_FOR_CHECKING)
#include <gsl/gsl_poly.h>
#endif
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>


namespace {
namespace local {

std::size_t iterations = 0;

struct my_constraint_data
{
	double a, b;
};

double my_objective_func(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
	++iterations;

	if (!grad.empty())
	{
		grad[0] = 0.0;
		grad[1] = 0.5 / std::sqrt(x[1]);
	}
	return std::sqrt(x[1]);
}

double my_constraint_func(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
	const my_constraint_data *d = reinterpret_cast<my_constraint_data *>(data);
	const double a = d->a, b = d->b;
	if (!grad.empty())
	{
		grad[0] = 3 * a * (a*x[0] + b) * (a*x[0] + b);
		grad[1] = -1.0;
	}
	return ((a*x[0] + b) * (a*x[0] + b) * (a*x[0] + b) - x[1]);
}

// REF [site] >> http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial
void simple_cpp_sample_using_gradient_based_algorithm()
{
	// Algorithm and dimensionality.
	nlopt::opt opt(nlopt::LD_MMA, 2);

	// Note that we do not need to set an upper bound (set_upper_bounds), since we are happy with the default upper bounds (+inf).
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL;
	lb[1] = 0.0;
	opt.set_lower_bounds(lb);

	opt.set_min_objective(my_objective_func, NULL);

	const my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	const double tol = 1e-8;  // An optional tolerance for the constraint.
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[0], tol);
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[1], tol);

	// A relative tolerance on the optimization parameters.
	opt.set_xtol_rel(1e-4);

	// Some initial guess.
	std::vector<double> x(2);
	x[0] = 1.234;
	x[1] = 5.678;

	double minf;  // The minimum objective value, upon return.
	const nlopt::result result = opt.optimize(x, minf);
	if (result < 0)
		std::cerr << "NLopt failed!" << std::endl;
	else
	{
		std::cout << "Found minimum after " << iterations << " evaluations" << std::endl;
		std::cout << "Found minimum at f(" << x[0] << ", " << x[1] << ") = " << minf << std::endl;
	}
}

// REF [site] >> http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial
void simple_cpp_sample_using_derivative_free_algorithm()
{
	// Algorithm and dimensionality.
	nlopt::opt opt(nlopt::LN_COBYLA, 2);

	// Note that we do not need to set an upper bound (set_upper_bounds), since we are happy with the default upper bounds (+inf).
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL;
	lb[1] = 0.0;
	opt.set_lower_bounds(lb);

	opt.set_min_objective(my_objective_func, NULL);

	const my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	const double tol = 1e-8;  // An optional tolerance for the constraint.
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[0], tol);
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[1], tol);

	// A relative tolerance on the optimization parameters.
	opt.set_xtol_rel(1e-4);
	opt.set_stopval(std::sqrt(8. / 27.) + 1e-3);

	// Some initial guess.
	std::vector<double> x(2);
	x[0] = 1.234;
	x[1] = 5.678;

	double minf;  // The minimum objective value, upon return.
	const nlopt::result result = opt.optimize(x, minf);
	if (result < 0)
		std::cerr << "NLopt failed!" << std::endl;
	else
	{
		std::cout << "Found minimum after " << iterations << " evaluations" << std::endl;
		std::cout << "Found minimum at f(" << x[0] << ", " << x[1] << ") = " << minf << std::endl;
	}
}

struct my_quadratic_data
{
	double a, b, c, d;  // Quadratic curve equation: a * x^2 + b * x + c * y + d = 0.
	double x0, y0;  // A sample point.
	size_t iterations;
};

double distance_with_quadratic_curve(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
{
	assert(grad.empty());

	my_quadratic_data* quad_data = reinterpret_cast<my_quadratic_data*>(my_func_data);
	assert(quad_data);

	++quad_data->iterations;

	const double eps = 1.0e-10;
	const double a = quad_data->a;
	const double b = quad_data->b;
	const double c = quad_data->c;
	const double d = quad_data->d;
	const double x0 = quad_data->x0;
	const double y0 = quad_data->y0;

	assert(std::abs(c) > eps);

	const double xf = x[0], yf = -(a * xf*xf + b * xf + d) / c;

	//return std::sqrt((xf - x0)*(xf - x0) + (yf - y0)*(yf - y0));
	return (xf - x0)*(xf - x0) + (yf - y0)*(yf - y0);
}

#if __USE_GSL_FOR_CHECKING
void min_distance_with_quadratic_curve_using_derivative_free_algorithm(const double a, const double b, const double c, const double d, const double x0, const double y0, size_t &errorCount)
#else
void min_distance_with_quadratic_curve_using_derivative_free_algorithm(const double a, const double b, const double c, const double d, const double x0, const double y0)
#endif
{
	// Algorithm and dimensionality.
	nlopt::opt opt(nlopt::LN_COBYLA, 1);

	my_quadratic_data quad_data;
	quad_data.a = a;
	quad_data.b = b;
	quad_data.c = c;
	quad_data.d = d;
	quad_data.x0 = x0;
	quad_data.y0 = y0;
	quad_data.iterations = 0;
	opt.set_min_objective(distance_with_quadratic_curve, (void *)&quad_data);

	// A relative tolerance on the optimization parameters.
	opt.set_xtol_rel(1e-4);
	opt.set_stopval(std::sqrt(8. / 27.) + 1e-3);

	// Some initial guess.
	std::vector<double> x(1);
	x[0] = 0.0;

	double minf;  // The minimum objective value, upon return.
	const nlopt::result result = opt.optimize(x, minf);
	if (result < 0)
		std::cerr << "NLopt failed!" << std::endl;
	else
	{
		std::cout << "Found minimum after " << quad_data.iterations << " evaluations" << std::endl;
		//std::cout << "Found minimum at f(" << x[0] << ") = " << minf << std::endl;
		std::cout << "Found minimum at f(" << x[0] << ") = " << std::sqrt(minf) << std::endl;

		const double xs = x[0], ys = -(a * xs*xs + b * xs + d) / c;
		std::cout << "Nearest point: " << "(" << xs << "," << ys << ")" << std::endl;
		std::cout << "Distance between (" << x0 << "," << y0 << ") and (" << xs << "," << ys << ") = " << std::sqrt((xs - x0)*(xs - x0) + (ys - y0)*(ys - y0)) << std::endl;

#if __USE_GSL_FOR_CHECKING
		// For checking.
		const double eps = 1.0e-10;
		const double c_2 = c * c;
		assert(c_2 > eps);
		const double aa = 4.0*a*a / c_2, bb = 6.0*a*b / c_2, cc = 2.0*(b*b / c_2 + 2.0*a*(d + c*y0) / c_2 + 1.0), dd = 2.0*(b*(d + c*y0) / c_2 - x0);
		assert(aa > eps);

		gsl_complex z[3];
		gsl_poly_complex_solve_cubic(bb / aa, cc / aa, dd / aa, &z[0], &z[1], &z[2]);

		int selected = -1;
		double minDist2 = std::numeric_limits<double>::max();
		for (int i = 0; i < 3; ++i)
			if (std::abs(z[i].dat[1]) < eps)
			{
				const double xx = z[i].dat[0], yy = -(a * xx*xx + b * xx + d) / c;
				const double dist2 = (xx - x0)*(xx - x0) + (yy - y0)*(yy - y0);

				if (dist2 < minDist2)
				{
					selected = i;
					minDist2 = dist2;
				}
			}
		assert(selected >= 0);

		const double xs2 = z[selected].dat[0], ys2 = -(a * xs2*xs2 + b * xs2 + d) / c;
		std::cout << "Distance between (" << x0 << "," << y0 << ") and (" << xs2 << "," << ys2 << ") = " << std::sqrt(minDist2) << std::endl;

		if (std::abs(xs - xs2) > 10e-4 || std::abs(ys - ys2) > 10e-4)
			++errorCount;
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_nlopt {

}  // namespace my_nlopt

int nlopt_main(int argc, char *argv[])
{
	// Tutorial ----------------------------------------------
	//local::simple_cpp_sample_using_gradient_based_algorithm();
	//local::simple_cpp_sample_using_derivative_free_algorithm();

	// -------------------------------------------------------
	{
		// Quadratic curve equation: x^2 - x + y - 2 = 0.
		const double a = 1.0, b = -1.0, c = 1.0, d = -2.0;

#if __USE_GSL_FOR_CHECKING
		std::random_device seedDevice;
		std::mt19937 RNG = std::mt19937(seedDevice());

		std::uniform_real_distribution<double> unifDist(-5, 5);  // [-5, 5].
		size_t iterations = 100;
		size_t errorCount = 0;
		for (int i = 0; i < iterations; ++i)
		{
			const double x0 = unifDist(RNG), y0 = unifDist(RNG);
			local::min_distance_with_quadratic_curve_using_derivative_free_algorithm(a, b, c, d, x0, y0, errorCount);
		}

		std::cout << "#errors = " << errorCount << ", #iterations = " << iterations << ", error rate = " << (double)errorCount / iterations << std::endl;
#else
		//const double x0 = 0.0, y0 = 0.0;
		//const double x0 = 5.0, y0 = 0.0;
		const double x0 = -5.0, y0 = 0.0;
		local::min_distance_with_quadratic_curve_using_derivative_free_algorithm(a, b, c, d, x0, y0);
#endif
	}

    return 0;
}
