#include <dlib/optimization.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

typedef dlib::matrix<double, 0, 1> column_vector;

double rosen(const column_vector& m)
{
	const double x = m(0);
	const double y = m(1);

	// compute Rosenbrock's function and return the result.
	return 100.0*std::pow(y - x*x, 2) + std::pow(1 - x, 2);
}

// Compute the gradient vector for the rosen() function.
const column_vector rosen_derivative(const column_vector& m)
{
	const double x = m(0);
	const double y = m(1);

	// Make us a column vector of length 2.
	column_vector res(2);

	// Compute the gradient vector.
	res(0) = -400 * x*(y - x*x) - 2 * (1 - x);  // Derivative of rosen() with respect to x.
	res(1) = 200 * (y - x*x);  // Derivative of rosen() with respect to y.
	return res;
}

// Compute the Hessian matrix for the rosen() fuction.
dlib::matrix<double> rosen_hessian(const column_vector& m)
{
	const double x = m(0);
	const double y = m(1);

	dlib::matrix<double> res(2, 2);

	// Compute the second derivatives.
	res(0, 0) = 1200 * x*x - 400 * y + 2;  // Second derivative with respect to x.
	res(1, 0) = res(0, 1) = -400 * x;  // Derivative with respect to x and y.
	res(1, 1) = 200;  // Second derivative with respect to y.
	return res;
}

class test_function
{
public:
	test_function(const column_vector& input)
	{
		target = input;
	}

	double operator() (const column_vector& arg) const
	{
		// Return the mean squared error between the target vector and the input vector.
		return mean(squared(target - arg));
	}

private:
	column_vector target;
};

class rosen_model
{
public:
	typedef column_vector column_vector;
	typedef dlib::matrix<double> general_matrix;

public:
	double operator() (const column_vector& x) const
	{
		return rosen(x);
	}

	void get_derivative_and_hessian(const column_vector& x, column_vector& der, general_matrix& hess) const
	{
		der = rosen_derivative(x);
		hess = rosen_hessian(x);
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

// REF [file] >> ${DLIB_HOME}/examples/optimization_ex.cpp
void optimization_example()
{
	// Make a column vector of length 2.
	local::column_vector starting_point(2);
	starting_point = 4, 8;

	std::cout << "Difference between analytic derivative and numerical approximation of derivative: "
		<< dlib::length(derivative(local::rosen)(starting_point) - local::rosen_derivative(starting_point)) << std::endl;

	// Use the BFGS algorithm.
	{
		// Use the find_min() function to find the minimum point.
		std::cout << "Find the minimum of the rosen function()." << std::endl;
		dlib::find_min(dlib::bfgs_search_strategy(),  // Use BFGS search algorithm.
			dlib::objective_delta_stop_strategy(1e-7),  // Stop when the change in rosen() is less than 1e-7.
			local::rosen, local::rosen_derivative, starting_point, -1);

		// Once the function ends the starting_point vector will contain the optimum point of (1,1).
		std::cout << "Rosen solution:\n" << starting_point << std::endl;

		// Let's try doing it again with a different starting point and the version of find_min() that doesn't require you to supply a derivative function.  
		starting_point = -94, 5.2;
		dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-7),
			local::rosen, starting_point, -1);

		// The correct minimum point is found and stored in starting_point.
		std::cout << "Rosen solution:\n" << starting_point << std::endl;
	}

	// Use the L-BFGS algorithm.
	// L-BFGS is very similar to the BFGS algorithm, however, BFGS uses O(N^2) memory where N is the size of the starting_point vector.  
	// The L-BFGS algorithm however uses only O(N) memory.
	{
		starting_point = 0.8, 1.3;
		dlib::find_min(dlib::lbfgs_search_strategy(10),  // The 10 here is basically a measure of how much memory L-BFGS will use.
			dlib::objective_delta_stop_strategy(1e-7).be_verbose(),  // Adding be_verbose() causes a message to be  printed for each iteration of optimization.
			local::rosen, local::rosen_derivative, starting_point, -1);

		std::cout << std::endl << "Rosen solution:\n" << starting_point << std::endl;

		starting_point = -94, 5.2;
		dlib::find_min_using_approximate_derivatives(dlib::lbfgs_search_strategy(10),
			dlib::objective_delta_stop_strategy(1e-7),
			local::rosen, starting_point, -1);

		std::cout << "Rosen solution:\n" << starting_point << std::endl;
	}

	// dlib also supports solving functions subject to bounds constraints on the variables.
	// So for example, if you wanted to find the minimizer of the rosen function where both input variables were in the range 0.1 to 0.8 you would do it like this:
	{
		starting_point = 0.1, 0.1;  // Start with a valid point inside the constraint box.
		dlib::find_min_box_constrained(dlib::lbfgs_search_strategy(10),
			dlib::objective_delta_stop_strategy(1e-9),
			local::rosen, local::rosen_derivative, starting_point, 0.1, 0.8);

		std::cout << std::endl << "Constrained rosen solution:\n" << starting_point << std::endl;

		// Use an approximate derivative.
		starting_point = 0.1, 0.1;
		dlib::find_min_box_constrained(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-9),
			local::rosen, dlib::derivative(local::rosen), starting_point, 0.1, 0.8);

		std::cout << std::endl << "Constrained rosen solution:\n" << starting_point << std::endl;
	}

	// Provide second derivative information to the optimizers.
	{
		starting_point = 0.8, 1.3;
		dlib::find_min(dlib::newton_search_strategy(local::rosen_hessian),
			dlib::objective_delta_stop_strategy(1e-7),
			local::rosen, local::rosen_derivative, starting_point, -1);

		std::cout << "Rosen solution:\n" << starting_point << std::endl;

		// Use find_min_trust_region(), which is also a method which uses second derivatives.
		// For some kinds of non-convex function it may be more reliable than using a newton_search_strategy with find_min().
		starting_point = 0.8, 1.3;
		dlib::find_min_trust_region(dlib::objective_delta_stop_strategy(1e-7),
			local::rosen_model(),
			starting_point,
			10  // Initial trust region radius.
		);

		std::cout << "Rosen solution:\n" << starting_point << std::endl;
	}

	// Use the test_function object with the optimization functions.
	{
		std::cout << "\nFind the minimum of the test_function" << std::endl;

		local::column_vector target(4);
		starting_point.set_size(4);

		// This variable will be used as the target of the test_function.
		// So, our simple test_function object will have a global minimum at the point given by the target.
		// We will then use the optimization routines to find this minimum value.
		target = 3, 5, 1, 7;

		// Set the starting point far from the global minimum.
		starting_point = 1, 2, 3, 4;
		dlib::find_min_using_approximate_derivatives(dlib::bfgs_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-7),
			local::test_function(target), starting_point, -1);

		// At this point the correct value of (3,5,1,7) should be found and stored in starting_point.
		std::cout << "test_function solution:\n" << starting_point << std::endl;

		// Use the conjugate gradient algorithm.
		starting_point = -4, 5, 99, 3;
		dlib::find_min_using_approximate_derivatives(dlib::cg_search_strategy(),
			dlib::objective_delta_stop_strategy(1e-7),
			local::test_function(target), starting_point, -1);

		std::cout << "test_function solution:\n" << starting_point << std::endl;

		// Use the BOBYQA algorithm.
		// This is a technique specially designed to minimize a function in the absence of derivative information.  
		starting_point = -4, 5, 99, 3;
		dlib::find_min_bobyqa(local::test_function(target),
			starting_point,
			9,  // Number of interpolation points.
			dlib::uniform_matrix<double>(4, 1, -1e100),  // Lower bound constraint.
			dlib::uniform_matrix<double>(4, 1, 1e100),  // Upper bound constraint.
			10,  // Initial trust region radius.
			1e-6,  // Stopping trust region radius.
			100  // Max number of objective function evaluations.
		);

		std::cout << "test_function solution:\n" << starting_point << std::endl;
	}
}

}  // namespace my_dlib
