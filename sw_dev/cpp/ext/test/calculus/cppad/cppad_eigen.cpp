#define _SCL_SECURE_NO_WARNINGS 1
#include <cppad/example/cppad_eigen.hpp>
#include <cppad/speed/det_by_minor.hpp>
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

// REF [file] >> ${COIN-OR_HOME}/cppad/example/eigen_array.cpp
void eigen_array_example()
{
	typedef Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> cppad_vector_type;

	// Domain and range space vectors.
	const size_t n = 10, m = n;
	cppad_vector_type a_x(n), a_y(m);

	// Set and declare independent variables and start tape recording.
	for (size_t j = 0; j < n; ++j)
		a_x[j] = double(1 + j);
	CppAD::Independent(a_x);

	// Evaluate a component wise function.
	a_y = a_x.array() + Eigen::sin(a_x.array());

	// Create f: x -> y and stop tape recording.
	CppAD::ADFun<double> f(a_x, a_y);

	// Compute the derivative of y w.r.t x using CppAD.
	std::vector<double> x(n);
	for (size_t j = 0; j < n; ++j)
		x[j] = double(j) + 1.0 / double(j + 1);
	const std::vector<double> &jacob = f.Jacobian(x);

	//const double eps = 100. * CppAD::numeric_limits<double>::epsilon();
	std::cout << "The Jacobian of x + sin(x) = " << std::endl;
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
			std::cout << jacob[i * n + j] << ", ";
		std::cout << std::endl;
	}
	std::cout << "The true value of 1 + cos(x) = " << std::endl;
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < n; ++j)
			std::cout << (i != j ? 0.0 : 1.0 + std::cos(x[i])) << ", ";
		std::cout << std::endl;
	}
}

// REF [file] >> ${COIN-OR_HOME}/cppad/example/eigen_det.cpp
void eigen_det_example()
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_type;
	typedef Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, Eigen::Dynamic> cppad_matrix_type;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_type;
	typedef Eigen::Matrix<CppAD::AD<double>, Eigen::Dynamic, 1> cppad_vector_type;

	// Domain and range space vectors.
	const size_t size = 3, n = size * size, m = 1;
	cppad_vector_type a_x(n), a_y(m);
	vector_type x(n);

	// Set and declare independent variables and start tape recording.
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
		{
			// Lower triangular matrix.
			a_x[i * size + j] = x[i * size + j] = 0.0;
			if (j <= i)
				a_x[i * size + j] = x[i * size + j] = double(1 + i + j);
		}

	CppAD::Independent(a_x);

	// Copy independent variable vector to a matrix.
	cppad_matrix_type a_X(size, size);
	matrix_type X(size, size);
	for (size_t i = 0; i < size; ++i)
		for (size_t j = 0; j < size; ++j)
		{
			X(i, j) = x[i * size + j];
			// If we used a_X(i, j) = X(i, j), a_X would not depend on a_x.
			a_X(i, j) = a_x[i * size + j];
		}

	// Compute the log of determinant of X.
	a_y[0] = CppAD::log(a_X.determinant());

	// Create f: x -> y and stop tape recording.
	CppAD::ADFun<double> f(a_x, a_y);

	//const double eps = 100.0 * CppAD::numeric_limits<double>::epsilon();
	CppAD::det_by_minor<double> det(size);
	std::cout << "log(det(X)) computed by CppAD = " << CppAD::Value(a_y[0]) << ", the true value = " << std::log(det(x)) << std::endl;

	// Compute the derivative of y w.r.t x using CppAD.
	const vector_type &jac = f.Jacobian(x);

	// d/dX log(det(X)) = transpose(inv(X)).
	std::cout << "The Jacobian of log(det(X)) = " << std::endl;
	for (size_t i = 0; i < size; ++i)
	{
		for (size_t j = 0; j < size; ++j)
			std::cout << jac[i * size + j] << ", ";
		std::cout << std::endl;
	}
	const matrix_type invX = X.inverse();
	std::cout << "The true value of d/dX log(det(X)) = " << std::endl;
	for (size_t i = 0; i < size; ++i)
	{
		for (size_t j = 0; j < size; ++j)
			std::cout << invX(j, i) << ", ";
		std::cout << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_cppad {

void eigen()
{
	local::eigen_array_example();
	local::eigen_det_example();
}

}  // namespace my_cppad

#undef _SCL_SECURE_NO_WARNINGS
