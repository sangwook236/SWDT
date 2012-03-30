//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void svd()
{
	const size_t nrow = 5, ncol = 3;
	typedef Eigen::Matrix<double, nrow, ncol> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	// MxN matrix, K=min(M,N), M>=N
	//const Eigen::SVD<MatrixType> svd(m);
	const Eigen::JacobiSVD<MatrixType> svd = m.jacobiSvd();
	std::cout << "singular value decomposition:" << std::endl;

	// K vector
	std::cout << "singular values:" << std::endl;
	const Eigen::Matrix<double, ncol, 1> &sigmas = svd.singularValues();
	std::cout << sigmas << std::endl;
	std::cout << "singular value matrix:" << std::endl;
	const Eigen::DiagonalMatrix<double, ncol> &S = sigmas.asDiagonal();
	//const Eigen::DiagonalMatrix<Eigen::VectorXd> &S = sigmas.asDiagonal();  // error !!!
	std::cout << Eigen::Matrix<double, ncol, ncol>(S) << std::endl;

	// MxK matrix
	std::cout << "left singular vectors:" << std::endl;
	const Eigen::JacobiSVD<MatrixType>::MatrixUType &U = svd.matrixU();
	std::cout << U << std::endl;

	// KxN matrix
	std::cout << "right singular vectors:" << std::endl;
	const Eigen::JacobiSVD<MatrixType>::MatrixVType &V = svd.matrixV();
	std::cout << V << std::endl;

	// FIXME [correct] >> compile-time error
/*
	std::cout << "reconstruct the original matrix m:" << std::endl;
	std::cout << U * S * V.transpose() << std::endl;
*/
}
