//#include "stdafx.h"
#include <nlopt.hpp>
#include <iostream>
#include <vector>
#include <cmath>


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

// [ref] http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial
void simple_cpp_sample_using_gradient_based_algorithm()
{
	// algorithm and dimensionality
	nlopt::opt opt(nlopt::LD_MMA, 2);

	// Note that we do not need to set an upper bound (set_upper_bounds), since we are happy with the default upper bounds (+inf).
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL;
	lb[1] = 0.0;
	opt.set_lower_bounds(lb);

	opt.set_min_objective(my_objective_func, NULL);

	const my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	const double tol = 1e-8;  // an optional tolerance for the constraint.
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[0], tol);
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[1], tol);

	// a relative tolerance on the optimization parameters.
	opt.set_xtol_rel(1e-4);

	// some initial guess.
	std::vector<double> x(2);
	x[0] = 1.234;
	x[1] = 5.678;

	double minf;  // the minimum objective value, upon return.
	const nlopt::result result = opt.optimize(x, minf);
	if (result < 0)
		std::cerr << "nlopt failed!" << std::endl;
	else
	{
		std::cout << "found minimum after " << iterations << " evaluations" << std::endl;
		std::cout << "found minimum at f(" << x[0] << ", " << x[1] << ") = " << minf << std::endl;
	}
}

// [ref] http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial
void simple_cpp_sample_using_derivative_free_algorithm()
{
	// algorithm and dimensionality
	nlopt::opt opt(nlopt::LN_COBYLA, 2);

	// Note that we do not need to set an upper bound (set_upper_bounds), since we are happy with the default upper bounds (+inf).
	std::vector<double> lb(2);
	lb[0] = -HUGE_VAL;
	lb[1] = 0.0;
	opt.set_lower_bounds(lb);

	opt.set_min_objective(my_objective_func, NULL);

	const my_constraint_data data[2] = { { 2, 0 }, { -1, 1 } };
	const double tol = 1e-8;  // an optional tolerance for the constraint.
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[0], tol);
	opt.add_inequality_constraint(my_constraint_func, (void *)&data[1], tol);

	// a relative tolerance on the optimization parameters.
	opt.set_xtol_rel(1e-4);
	opt.set_stopval(std::sqrt(8. / 27.) + 1e-3);

	// some initial guess.
	std::vector<double> x(2);
	x[0] = 1.234;
	x[1] = 5.678;

	double minf;  // the minimum objective value, upon return.
	const nlopt::result result = opt.optimize(x, minf);
	if (result < 0)
		std::cerr << "nlopt failed!" << std::endl;
	else
	{
		std::cout << "found minimum after " << iterations << " evaluations" << std::endl;
		std::cout << "found minimum at f(" << x[0] << ", " << x[1] << ") = " << minf << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_nlopt {

}  // namespace my_nlopt

int nlopt_main(int argc, char *argv[])
{
	local::simple_cpp_sample_using_gradient_based_algorithm();
	local::simple_cpp_sample_using_derivative_free_algorithm();

    return 0;
}
