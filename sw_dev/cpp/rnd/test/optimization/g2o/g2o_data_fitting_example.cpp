//#include "stdafx.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/auto_differentiation.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/stuff/sampler.h>
//#include <g2o/stuff/command_args.h>


namespace {
namespace local {

G2O_USE_OPTIMIZATION_LIBRARY(dense);

double errorOfSolution(int numPoints, const std::vector<Eigen::Vector2d>& points, const Eigen::Vector3d& circle)
{
	const Eigen::Vector2d& center = circle.head<2>();
	const double radius = circle(2);
	double error = 0.0;
	for (int i = 0; i < numPoints; ++i)
	{
		const double d = (points[i] - center).norm() - radius;
		error += d * d;
	}
	return error;
}

/**
 * \brief a circle located at x,y with radius r
 */
class VertexCircle : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	VertexCircle() {}

	bool read(std::istream& /*is*/) override { return false; }
	bool write(std::ostream& /*os*/) const override { return false; }

	void setToOriginImpl() override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
	}

	void oplusImpl(const double* update) override
	{
		Eigen::Vector3d::ConstMapType v(update);
		_estimate += v;
	}
};

/**
 * \brief measurement for a point on the circle
 *
 * Here the measurement is the point which is on the circle.
 * The error function computes the distance of the point to the center minus the radius of the circle.
 */
class EdgePointOnCircle : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexCircle>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	EdgePointOnCircle() {}

	bool read(std::istream& /*is*/) override { return false; }
	bool write(std::ostream& /*os*/) const override { return false; }

	template <typename T>
	bool operator()(const T* circle, T* error) const
	{
		typename g2o::VectorN<2, T>::ConstMapType center(circle);
		const T& radius = circle[2];

		error[0] = (measurement().cast<T>() - center).norm() - radius;
		return true;
	}

	G2O_MAKE_AUTO_AD_FUNCTIONS  // Use autodiff.
};

/**
 * \brief the params, a, b, and lambda for a * exp(-lambda * t) + b
 */
class VertexParams : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	VertexParams() {}

	bool read(std::istream& /*is*/) override { return false; }
	bool write(std::ostream& /*os*/) const override { return false; }

	void setToOriginImpl() override {}

	void oplusImpl(const double* update) override
	{
		const Eigen::Vector3d::ConstMapType v(update);
		_estimate += v;
	}
};

/**
 * \brief measurement for a point on the curve
 *
 * Here the measurement is the point which is lies on the curve.
 * The error function computes the difference between the curve and the point.
 */
class EdgePointOnCurve : public g2o::BaseUnaryEdge<1, Eigen::Vector2d, VertexParams>
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	EdgePointOnCurve() {}

	bool read(std::istream& /*is*/) override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}
	bool write(std::ostream& /*os*/) const override
	{
		std::cerr << __PRETTY_FUNCTION__ << " not implemented yet" << std::endl;
		return false;
	}

	template <typename T>
	bool operator()(const T* params, T* error) const
	{
		const T& a = params[0];
		const T& b = params[1];
		const T& lambda = params[2];
		T fval = a * exp(-lambda * T(measurement()(0))) + b;
		error[0] = fval - measurement()(1);
		return true;
	}

	G2O_MAKE_AUTO_AD_FUNCTIONS  // Use autodiff.
};

}  // namespace local
}  // unnamed namespace

namespace my_g2o {

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/data_fitting/circle_fit.cpp
void circle_fit_example()
{
	const int numPoints = 100;  // Number of points sampled from the circle.
	const int maxIterations = 10;  // Perform n iterations.
	const bool verbose = false;  // Verbose output of the optimization process.

	// Generate random data.
	const Eigen::Vector2d center(4.0, 2.0);
	const double radius = 2.0;
	std::vector<Eigen::Vector2d> points(numPoints);

	g2o::Sampler::seedRand();
	for (int i = 0; i < numPoints; ++i)
	{
		const double r = g2o::Sampler::gaussRand(radius, 0.05);
		const double angle = g2o::Sampler::uniformRand(0.0, 2.0 * M_PI);
		points[i].x() = center.x() + r * std::cos(angle);
		points[i].y() = center.y() + r * std::sin(angle);
	}

	// Setup the solver.
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

#if 1
	// Allocate the solver.
	g2o::OptimizationAlgorithmProperty solverProperty;
	optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solverProperty));
#else
	using MyBlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >;
	using MyLinearSolver = g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> ;

	auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<MyBlockSolver>(g2o::make_unique<MyLinearSolver>()));
	optimizer.setAlgorithm(solver);
#endif

	// Build the optimization problem given the points.
	// 1. add the circle vertex.
	auto circle = new local::VertexCircle();
	circle->setId(0);
	circle->setEstimate(Eigen::Vector3d(3.0, 3.0, 3.0));  // Some initial value for the circle.
	optimizer.addVertex(circle);
	// 2. add the points we measured.
	for (int i = 0; i < numPoints; ++i)
	{
		auto e = new local::EdgePointOnCircle;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		e->setVertex(0, circle);
		e->setMeasurement(points[i]);
		optimizer.addEdge(e);
	}

	// Perform the optimization.
	optimizer.initializeOptimization();
	optimizer.setVerbose(verbose);
	optimizer.optimize(maxIterations);

	if (verbose) std::cout << std::endl;

	// Print out the result.
	std::cout << "Iterative least squares solution" << std::endl;
	std::cout << "center of the circle " << circle->estimate().head<2>().transpose() << std::endl;
	std::cout << "radius of the cirlce " << circle->estimate()(2) << std::endl;
	std::cout << "error " << local::errorOfSolution(numPoints, points, circle->estimate()) << std::endl;
	std::cout << std::endl;

	// Solve by linear least squares.
	// Let (a, b) be the center of the circle and r the radius of the circle.
	// For a point (x, y) on the circle we have:
	// (x - a)^2 + (y - b)^2 = r^2
	// This leads to
	// (-2x -2y 1)^T * (a b c) = -x^2 - y^2   (1)
	// where c = a^2 + b^2 - r^2.
	// Since we have a bunch of points, we accumulate Eqn (1) in a matrix and
	// compute the normal equation to obtain a solution for (a b c).
	// Afterwards the radius r is recovered.
	Eigen::MatrixXd A(numPoints, 3);
	Eigen::VectorXd b(numPoints);
	for (int i = 0; i < numPoints; ++i)
	{
		A(i, 0) = -2 * points[i].x();
		A(i, 1) = -2 * points[i].y();
		A(i, 2) = 1;
		b(i) = -std::pow(points[i].x(), 2) - std::pow(points[i].y(), 2);
	}
	Eigen::Vector3d solution = (A.transpose() * A).ldlt().solve(A.transpose() * b);
	// Calculate the radius of the circle given the solution so far.
	solution(2) = std::sqrt(std::pow(solution(0), 2) + std::pow(solution(1), 2) - solution(2));
	std::cout << "Linear least squares solution" << std::endl;
	std::cout << "center of the circle " << solution.head<2>().transpose() << std::endl;
	std::cout << "radius of the cirlce " << solution(2) << std::endl;
	std::cout << "error " << local::errorOfSolution(numPoints, points, solution) << std::endl;
}

// REF [site] >> https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/data_fitting/curve_fit.cpp
void curve_fit_example()
{
	const int numPoints = 50;  // Number of points sampled from the curve.
	const int maxIterations = 10;  // Perform n iterations.
	const bool verbose = false;  // Verbose output of the optimization process.
	std::string dumpFilename;  // Dump the points into a file.

	// Generate random data.
	g2o::Sampler::seedRand();
	const double a = 2.0;
	const double b = 0.4;
	const double lambda = 0.2;
	std::vector<Eigen::Vector2d> points(numPoints);
	for (int i = 0; i < numPoints; ++i)
	{
		const double x = g2o::Sampler::uniformRand(0, 10);
		const double y = a * std::exp(-lambda * x) + b + g2o::Sampler::gaussRand(0, 0.02);  // Add Gaussian noise.
		points[i].x() = x;
		points[i].y() = y;
	}

	if (dumpFilename.size() > 0)
	{
		std::ofstream fout(dumpFilename.c_str());
		for (int i = 0; i < numPoints; ++i) fout << points[i].transpose() << std::endl;
	}

	// Setup the solver.
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	// Allocate the solver.
	g2o::OptimizationAlgorithmProperty solverProperty;
	optimizer.setAlgorithm(g2o::OptimizationAlgorithmFactory::instance()->construct("lm_dense", solverProperty));

	// Build the optimization problem given the points.
	// 1. add the parameter vertex.
	auto params = new local::VertexParams();
	params->setId(0);
	params->setEstimate(Eigen::Vector3d(1, 1, 1));  // Some initial value for the params.
	optimizer.addVertex(params);
	// 2. add the points we measured to be on the curve.
	for (int i = 0; i < numPoints; ++i)
	{
		auto e = new local::EdgePointOnCurve;
		e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
		e->setVertex(0, params);
		e->setMeasurement(points[i]);
		optimizer.addEdge(e);
	}

	// Perform the optimization.
	optimizer.initializeOptimization();
	optimizer.setVerbose(verbose);
	optimizer.optimize(maxIterations);

	if (verbose) std::cout << std::endl;

	// Print out the result.
	std::cout << "Target curve" << std::endl;
	std::cout << "a * exp(-lambda * x) + b" << std::endl;
	std::cout << "Iterative least squares solution" << std::endl;
	std::cout << "a      = " << params->estimate()(0) << std::endl;
	std::cout << "b      = " << params->estimate()(1) << std::endl;
	std::cout << "lambda = " << params->estimate()(2) << std::endl;
	std::cout << std::endl;
}

}  // namespace my_g2o
