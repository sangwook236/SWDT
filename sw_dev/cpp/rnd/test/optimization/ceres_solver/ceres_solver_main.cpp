//#include "stdafx.h"
#include <cmath>
#include <iostream>
#include <glog/logging.h>
#include <ceres/ceres.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES 1


namespace {
namespace local {

struct CostFunctor
{
	template <typename T>
	bool operator()(const T * const x, T *residual) const
	{
		// f(x) = 10 − x.
		residual[0] = T(10.0) - x[0];
		return true;
	}
};

// REF [site] >> http://ceres-solver.org/nnls_tutorial.html
void auto_differentiation_example()
{
	// The variable to solve for with its initial value.
	const double initial_x = 0.5;
	double x = initial_x;

	// Build the problem.
	ceres::Problem problem;

	// Set up the only cost function (also known as residual).
	// Use auto-differentiation to obtain the derivative (jacobian).
	ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
	problem.AddResidualBlock(cost_function, nullptr, &x);

	// Run the solver.
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << "x : " << initial_x << " -> " << x << std::endl;
}

struct NumericDiffCostFunctor
{
	bool operator()(const double * const x, double *residual) const
	{
		residual[0] = 10.0 - x[0];
		return true;
	}
};

// REF [site] >> http://ceres-solver.org/nnls_tutorial.html
void numeric_differentiation_example()
{
	// The variable to solve for with its initial value.
	const double initial_x = 0.5;
	double x = initial_x;

	// Build the problem.
	ceres::Problem problem;

	// Set up the only cost function (also known as residual).
	// Use numeric differentiation to obtain the derivative (jacobian).
	ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(new NumericDiffCostFunctor);
	//ceres::CostFunction *cost_function = new ceres::NumericDiffCostFunction<CostFunctor, ceres::CENTRAL, 1, 1>(new CostFunctor);
	problem.AddResidualBlock(cost_function, nullptr, &x);

	// Run the solver.
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << "x : " << initial_x << " -> " << x << std::endl;
}

// A CostFunction implementing analytically derivatives for the
// function f(x) = 10 - x.
class QuadraticCostFunction : public ceres::SizedCostFunction<1 /* number of residuals */, 1 /* size of first parameter */>
{
public:
	virtual ~QuadraticCostFunction()
	{}

public:
	virtual bool Evaluate(double const * const *parameters, double *residuals, double **jacobians) const
	{
		// f(x) = 10 - x.
		const double x = parameters[0][0];
		residuals[0] = 10 - x;

		// Compute the Jacobian if asked for.
		// f'(x) = -1.
		// Since there's only 1 parameter and that parameter has 1 dimension, there is only 1 element to fill in the jacobians.
		if (nullptr != jacobians && nullptr != jacobians[0])
		{
			jacobians[0][0] = -1;
		}

		return true;
	}
};

// REF [site] >> http://ceres-solver.org/nnls_tutorial.html
void analytic_differentiation_example()
{
	// The variable to solve for with its initial value.
	const double initial_x = 0.5;
	double x = initial_x;

	// Build the problem.
	ceres::Problem problem;

	// Set up the only cost function (also known as residual).
	ceres::CostFunction *cost_function = new QuadraticCostFunction;
	problem.AddResidualBlock(cost_function, nullptr /* squared loss */, &x);

	// Run the solver.
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.BriefReport() << std::endl;
	std::cout << "x : " << initial_x << " -> " << x << std::endl;
}

struct Powell_F1
{
	template <typename T>
	bool operator()(const T * const x1, const T * const x2, T *residual) const
	{
		// f1 = x1 + 10 * x2;
		residual[0] = x1[0] + T(10.0) * x2[0];
		return true;
	}
};

struct Powell_F2
{
	template <typename T>
	bool operator()(const T * const x3, const T * const x4, T *residual) const
	{
		// f2 = sqrt(5) * (x3 - x4)
		residual[0] = T(sqrt(5.0)) * (x3[0] - x4[0]);
		return true;
	}
};

struct Powell_F3
{
	template <typename T>
	bool operator()(const T * const x2, const T * const x4, T *residual) const
	{
		// f3 = (x2 - 2 *   x3)^2
		residual[0] = (x2[0] - T(2.0) * x4[0]) * (x2[0] - T(2.0) * x4[0]);
		return true;
	}
};

struct Powell_F4
{
	template <typename T>
	bool operator()(const T * const x1, const T * const x4, T *residual) const
	{
		// f4 = sqrt(10) * (x1 - x4)^2
		residual[0] = T(sqrt(10.0)) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
		return true;
	}
};

DEFINE_string(minimizer, "trust_region", "Minimizer type to use, choices are: line_search & trust_region");

// REF [site] >> http://ceres-solver.org/nnls_tutorial.html
/*
	Minimization of Powell's singular function.

	F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)

	f1 = x1 + 10 * x2;
	f2 = sqrt(5) * (x3 - x4)
	f3 = (x2 - 2 * x3)^2
	f4 = sqrt(10) * (x1 - x4)^2

	The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
	The minimum is 0 at (x1, x2, x3, x4) = 0.
*/
void Powells_function_example()
{
	// The variable to solve for with its initial value.
	double x1 =  3.0; double x2 = -1.0; double x3 =  0.0; double x4 = 1.0;

	// Build the problem.
	ceres::Problem problem;

	// Add residual terms to the problem using the autodiff wrapper to get the derivatives automatically.
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Powell_F1, 1, 1, 1>(new Powell_F1), nullptr, &x1, &x2);
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Powell_F2, 1, 1, 1>(new Powell_F2), nullptr, &x3, &x4);
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Powell_F3, 1, 1, 1>(new Powell_F3), nullptr, &x2, &x3);
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Powell_F4, 1, 1, 1>(new Powell_F4), nullptr, &x1, &x4);

	ceres::Solver::Options options;
	LOG_IF(FATAL, !ceres::StringToMinimizerType(FLAGS_minimizer, &options.minimizer_type)) << "Invalid minimizer: " << FLAGS_minimizer << ", valid options are: trust_region and line_search.";
	//const std::string minimizer("trust_region");  // {line_search, trust_region}.
	//ceres::StringToMinimizerType(minimizer, &options.minimizer_type);

	options.max_num_iterations = 100;
	options.linear_solver_type = ceres::DENSE_QR;
	options.minimizer_progress_to_stdout = true;

	std::cout << "Initial x1 = " << x1  << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4 << std::endl;

	// Run the solver.
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	std::cout << summary.FullReport() << std::endl;
	std::cout << "Final x1 = " << x1 << ", x2 = " << x2 << ", x3 = " << x3 << ", x4 = " << x4 << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_ceres_solver {

void curve_fitting_example();
void robust_curve_fitting_example();
void simple_bundle_adjustment_example();
void bundle_adjustment_example();

}  // namespace my_ceres_solver

int ceres_solver_main(int argc, char *argv[])
{
	//google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);

	//local::auto_differentiation_example();
	//local::numeric_differentiation_example();
	//local::analytic_differentiation_example();

	// Powell's function.
	//local::Powells_function_example();

	// Curve fitting.
	//my_ceres_solver::curve_fitting_example();
	//my_ceres_solver::robust_curve_fitting_example();

	// Bundle adjustment.
	//my_ceres_solver::simple_bundle_adjustment_example();
	my_ceres_solver::bundle_adjustment_example();

	// Other examples.
	// REF [site] >> http://ceres-solver.org/nnls_tutorial.html

	return 0;
}
