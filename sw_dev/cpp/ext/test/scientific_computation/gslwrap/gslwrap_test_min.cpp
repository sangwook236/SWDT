//#include "stdafx.h"
#include <gslwrap/min_fminimizer.h>
#include <gslwrap/multimin_fdfminimizer.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

struct my_f: public gsl::min_f
{
	virtual double operator()(const double &x)
	{
		return (a * x + b) * x + c;
	}

	double a;
	double b;
	double c;
};

struct fn1: public gsl::min_f
{
	virtual double operator()(const double &x)
	{
		return std::cos(x) + 1.0;
	}
};

struct my_function: public gsl::multimin_fdf<my_function>
{
	my_function() : multimin_fdf(this)  {}

	virtual double operator()(const gsl::vector &x)
	{
		return 10.0 * (x[0] - a) * (x[0] - a) + 20.0 * (x[1] - b) * (x[1] - b) + 30.0;
	}

	virtual void derivative(const gsl::vector &x, gsl::vector &g)
	{
		g[0] = 20.0 * (x[0] - a);
		g[1] = 40.0 * (x[1] - b);
	}

	double a;
	double b;
};

}  // namespace local
}  // unnamed namespace

namespace my_gslwrap {

void OneDimMinimiserTest()
{
	local::fn1 f;
	//gsl::min_fminimizer m(gsl_min_fminimizer_goldensection);
	gsl::min_fminimizer m;
	double mm = 2;
	double a = 0;
	double b = 6;
	m.set(f, mm, a, b);
	int status;
	do
	{
		status = m.iterate();
		mm = m.minimum();
		a = m.x_lower();
		b = m.x_upper();

		status = gsl_min_test_interval(a, b, 0.001, 0.0);

		if (GSL_SUCCESS == status)
			std::cout << "converged:" << std::endl;

		std::cout << m.GetNIterations() << " [" << a << ", " << b << "]  " << mm << ", " << M_PI << ", " << (mm - M_PI) << ", " << (b - a) << std::endl;

	} while (GSL_SUCCESS != status && !m.is_converged());
}

/*
using namespace::std;

void MultDimMinimiserTest2()
{
	my_function f;
	f.a = 1.0;
	f.b = 2.0;
	gsl::vector x(2);
	x[0] = 1;
	x[1] = 2;
	std::cout << "value at f(x) x=" << x << std::endl;
	std::cout << f(x) << endl;

	std::cout << "gradient at x=" << x << std::endl;
	gsl::vector g(2);
	f.derivative(x, g);
	std::cout << g << std::endl;
}
*/

void MultDimMinimiserTest()
{
	local::my_function f;
	unsigned int dim = 2;
	f.a = 1.0;
	f.b = 2.0;
	gsl::multimin_fdfminimizer mm(dim);

	// starting point
	gsl::vector x(dim);
	x[0] = 5.0;
	x[1] = 7.0;

	mm.set(f, x, 0.01, 1e-4);
	int status;
	unsigned int iter = 0;
	do
	{
		++iter;
		status = mm.iterate();

		if (status)
			break;

		status = gsl_multimin_test_gradient(mm.gradient().gslobj(), 1e-3);

		if (GSL_SUCCESS == status)
			std::cout << "minimum found at: " << std::endl;

		std::cout << iter << ", " << mm.x_value()[0] << ", " << mm.x_value()[1] << ", " << mm.minimum() << std::endl;
	} while (GSL_CONTINUE == status && iter < 100);
}

}  // namespace my_gslwrap
