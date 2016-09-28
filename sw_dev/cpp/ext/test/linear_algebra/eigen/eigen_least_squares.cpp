//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// Using the SVD decomposition.
void use_svd()
{
	Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
	std::cout << "Here is the matrix A:\n" << A << std::endl;
	Eigen::VectorXf b = Eigen::VectorXf::Random(3);
	std::cout << "Here is the right hand side b:\n" << b << std::endl;
	std::cout << "The least-squares solution is:\n" << A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
}

// Using the QR decomposition.
void use_qr()
{
	Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
	Eigen::VectorXf b = Eigen::VectorXf::Random(3);
	std::cout << "The solution using the QR decomposition is:\n" << A.colPivHouseholderQr().solve(b) << std::endl;
}

// Using normal equations.
void use_normal_equation()
{
	Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
	Eigen::VectorXf b = Eigen::VectorXf::Random(3);
	std::cout << "The solution using normal equations is:\n" << (A.transpose() * A).ldlt().solve(A.transpose() * b) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

// REF [site] >> https://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html
void linear_least_squares()
{
	local::use_svd();
	local::use_qr();
	local::use_normal_equation();
}

// REF [site] >> https://en.wikipedia.org/wiki/Least_squares
void nonlinear_least_squares()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace my_eigen
