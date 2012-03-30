//#include "stdafx.h"
#include <Eigen/Eigen>
#include <Eigen/Array>
#include <Eigen/Core>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void evd()
{
	const size_t dim = 4;
	typedef Eigen::Matrix<double, dim, dim> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	const Eigen::EigenSolver<MatrixType> evd(m);
	std::cout << "eigen decomposition:" << std::endl;

	std::cout << "eigenvalues:" << std::endl;
	const Eigen::Matrix<std::complex<double>, dim, 1> &eigvals = evd.eigenvalues();
	//const Eigen::Matrix<std::complex<double>, dim, 1> &eigvals = m.eigenvalues();
	std::cout << eigvals << std::endl;
	const Eigen::DiagonalMatrix<Eigen::Matrix<std::complex<double>, dim, 1> > &D = eigvals.asDiagonal();
	//const Eigen::DiagonalMatrix<Eigen::VectorXcd> &D = eigvals.asDiagonal();  // error !!!
	std::cout << D << std::endl;
	std::cout << "pseudo-eigenvalues:" << std::endl;
	const Eigen::Matrix<double, dim, dim> &Dp = evd.pseudoEigenvalueMatrix();
	std::cout << Dp << std::endl;

	std::cout << "eigenvectors:" << std::endl;
	const Eigen::Matrix<std::complex<double>, dim, dim> &U = evd.eigenvectors();
	std::cout << U << std::endl;
	std::cout << "pseudo-eigenvectors:" << std::endl;
	const Eigen::Matrix<double, dim, dim> &Up = evd.pseudoEigenvectors();
	std::cout << Up << std::endl;

	//
	std::cout << "reconstruct the original matrix m:" << std::endl;
	std::cout << U * D * U.inverse() << std::endl;

	Eigen::Matrix<std::complex<double>, dim, dim> invU;
	U.computeInverse(&invU);
	std::cout << U * D * invU << std::endl;

	std::cout << Up * Dp * Up.inverse() << std::endl;
}
