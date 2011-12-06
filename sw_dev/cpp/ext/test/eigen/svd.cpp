#include "stdafx.h"
#include <Eigen/SVD>
#include <Eigen/Array>
#include <Eigen/Core>


void svd()
{
	const size_t nrow = 5, ncol = 3;
	typedef Eigen::Matrix<double, nrow, ncol> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	// MxN matrix, K=min(M,N), M>=N
	//const Eigen::SVD<MatrixType> svd(m);
	const Eigen::SVD<MatrixType> svd = m.svd();
	std::cout << "singular value decomposition:" << std::endl;

	// K vector
	std::cout << "singular values:" << std::endl;
	const Eigen::Matrix<double, ncol, 1> &sigmas = svd.singularValues();
	std::cout << sigmas << std::endl;
	std::cout << "singular value matrix:" << std::endl;
	const Eigen::DiagonalMatrix<Eigen::Matrix<double, ncol, 1> > &S = sigmas.asDiagonal();
	//const Eigen::DiagonalMatrix<Eigen::VectorXd> &S = sigmas.asDiagonal();  // error !!!
	std::cout << S << std::endl;

	// MxK matrix
	std::cout << "left singular vectors:" << std::endl;
	const Eigen::Matrix<double, nrow, ncol> &U = svd.matrixU();
	std::cout << U << std::endl;

	// KxN matrix
	std::cout << "right singular vectors:" << std::endl;
	const Eigen::Matrix<double, ncol, ncol> &V = svd.matrixV();
	std::cout << V << std::endl;

	//
	std::cout << "reconstruct the original matrix m:" << std::endl;
	std::cout << U * S * V.transpose() << std::endl;
}
