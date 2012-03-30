//#include "stdafx.h"
#include <Eigen/QR>
#include <Eigen/Array>
#include <Eigen/Core>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void qr()
{
	const size_t nrow = 5, ncol = 3;
	typedef Eigen::Matrix<double, nrow, ncol> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	// MxN matrix, K=min(M,N), M>=N
	//const Eigen::QR<MatrixType> qr(m);
	const Eigen::QR<MatrixType> qr = m.qr();
	std::cout << "QR decomposition:" << std::endl;

	// MxK matrix
	std::cout << "Q matrix:" << std::endl;
	const Eigen::Matrix<double, nrow, ncol> &Q = qr.matrixQ();
	std::cout << Q << std::endl;

	// KxN matrix
	std::cout << "R matrix:" << std::endl;
	const Eigen::Matrix<double, ncol, ncol> &R = qr.matrixR();
	std::cout << R << std::endl;

	//
	std::cout << "reconstruct the original matrix m:" << std::endl;
	std::cout << Q * R << std::endl;
}
