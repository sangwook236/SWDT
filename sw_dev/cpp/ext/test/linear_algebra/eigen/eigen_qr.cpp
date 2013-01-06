//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace eigen {

void qr()
{
	const size_t nrow = 5, ncol = 3;
	typedef Eigen::Matrix<double, nrow, ncol> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	// MxN matrix, K=min(M,N), M>=N
	//const Eigen::QR<MatrixType> qr(m);
#if 1
	const Eigen::ColPivHouseholderQR<MatrixType> qr = m.colPivHouseholderQr();
#elif 0
	const Eigen::FullPivHouseholderQR<MatrixType> qr = m.fullPivHouseholderQr();
#else
	const Eigen::HouseholderQR<MatrixType> qr = m.householderQr();
#endif
	std::cout << "QR decomposition:" << std::endl;

	//
	std::cout << "QR matrix:" << std::endl;
	const Eigen::Matrix<double, nrow, ncol> &QR = qr.matrixQR();
	std::cout << QR << std::endl;
}

}  // namespace eigen
