//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

void evd_1()
{
	const size_t dim = 4;
	typedef Eigen::Matrix<double, dim, dim> MatrixType;

	MatrixType m = MatrixType::Random();
	std::cout << "Matrix m:" << std::endl << m << std::endl;

	const Eigen::EigenSolver<MatrixType> evd(m);
	std::cout << "Eigen decomposition:" << std::endl;

	std::cout << "Eigenvalues:" << std::endl;
	const Eigen::Matrix<std::complex<double>, dim, 1> &eigvals = evd.eigenvalues();
	//const Eigen::Matrix<std::complex<double>, dim, 1> &eigvals = m.eigenvalues();
	std::cout << eigvals << std::endl;
	const Eigen::DiagonalMatrix<std::complex<double>, dim> &D = eigvals.asDiagonal();
	//const Eigen::DiagonalMatrix<Eigen::VectorXcd> &D = eigvals.asDiagonal();  // Error !!!
	std::cout << Eigen::Matrix<std::complex<double>, dim, dim>(D) << std::endl;
	std::cout << "Pseudo-eigenvalues:" << std::endl;
	const Eigen::Matrix<double, dim, dim> &Dp = evd.pseudoEigenvalueMatrix();
	std::cout << Dp << std::endl;

	std::cout << "Eigenvectors:" << std::endl;
	const Eigen::Matrix<std::complex<double>, dim, dim> &U = evd.eigenvectors();
	std::cout << U << std::endl;
	std::cout << "Pseudo-eigenvectors:" << std::endl;
	const Eigen::Matrix<double, dim, dim> &Up = evd.pseudoEigenvectors();
	std::cout << Up << std::endl;

	//
	std::cout << "Reconstruct the original matrix m:" << std::endl;
	std::cout << U * D * U.inverse() << std::endl;

	const Eigen::Matrix<std::complex<double>, dim, dim> invU = U.inverse();
	std::cout << U * D * invU << std::endl;

	std::cout << Up * Dp * Up.inverse() << std::endl;
}

void evd_2()
{
    Eigen::Matrix2f A;
    A << 1, 2, 2, 3;
    std::cout << "Here is the matrix A:\n" << A << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigensolver(A);
    if (eigensolver.info() != Eigen::Success)
    {
        std::cout << "Failed: eigen decomposition" << std::endl;
        return;
    }

    std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
    std::cout << "Here's a matrix whose columns are eigenvectors of A \n"
        << "Corresponding to these eigenvalues:\n" << eigensolver.eigenvectors() << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void evd()
{
    local::evd_1();
    local::evd_2();
}

}  // namespace my_eigen
