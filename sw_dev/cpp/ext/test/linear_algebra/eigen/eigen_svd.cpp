//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void svd()
{
	const size_t nrow = 5, ncol = 3;
	typedef Eigen::Matrix<double, nrow, ncol> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "Matrix m:" << std::endl << m << std::endl;

	// MxN matrix, K=min(M,N), M>=N.
	//const Eigen::SVD<MatrixType> svd(m);
	const Eigen::JacobiSVD<MatrixType> svd = m.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << "Singular value decomposition:" << std::endl;

	// Singular values: K vector.
	std::cout << "Singular values:" << std::endl;
	const Eigen::Matrix<double, ncol, 1> &sigmas = svd.singularValues();
	std::cout << sigmas << std::endl;
	std::cout << "Singular value matrix:" << std::endl;
	const Eigen::DiagonalMatrix<double, ncol> &S = sigmas.asDiagonal();
	//const Eigen::DiagonalMatrix<Eigen::VectorXd> &S = sigmas.asDiagonal();  // Error !!!
	std::cout << Eigen::Matrix<double, ncol, ncol>(S) << std::endl;

	// Left singular vectors: MxK matrix.
	std::cout << "Left singular vectors:" << std::endl;
	const Eigen::JacobiSVD<MatrixType>::MatrixUType &U = svd.matrixU();
	std::cout << U << std::endl;

	// Right singular vectors: KxN matrix.
	std::cout << "Right singular vectors:" << std::endl;
	const Eigen::JacobiSVD<MatrixType>::MatrixVType &V = svd.matrixV();
	std::cout << V << std::endl;

	// Reconstruct.
	std::cout << "Reconstruct the original matrix m:" << std::endl;
	std::cout << U * S * V << std::endl;
}

}  // namespace my_eigen
