//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

void llt()
{
	const size_t dim = 4;
	typedef Eigen::Matrix<double, dim, dim> MatrixType;

	MatrixType m = MatrixType::Random();
	//m += m.transpose();  // error !!!
	MatrixType m2 = m + m.transpose();
	m = m2 * m2;
	std::cout << "matrix m:" << std::endl << m << std::endl;


	//const Eigen::LLT<MatrixType> llt(m);
	const Eigen::LLT<MatrixType> llt = m.llt();
	std::cout << "Cholesky decomposition: L * L^T:" << std::endl;
	//if (llt.isPositiveDefinite())  // not supported in EIGEN3
	{
		std::cout << "L matrix:" << std::endl;
		const Eigen::Matrix<double, dim, dim> &L = llt.matrixL();
		std::cout << L << std::endl;

		//
		std::cout << "reconstruct the original matrix m:" << std::endl;
		std::cout << L * L.transpose() << std::endl;
	}
	//else
	//	std::cout << "the matrix is not a positive definite" << std::endl;
}

void ldlt()
{
	const size_t dim = 4;
	typedef Eigen::Matrix<double, dim, dim> MatrixType;

	MatrixType m = MatrixType::Random();
	//m += m.transpose();  // error !!!
	MatrixType m2 = m + m.transpose();
	m = m2 * m2;
	std::cout << "matrix m:" << std::endl << m << std::endl;

	//const Eigen::LDLT<MatrixType> ldlt(m);
	const Eigen::LDLT<MatrixType> ldlt = m.ldlt();
	std::cout << "Cholesky decomposition: L * D * L^T:" << std::endl;
	//if (ldlt.isPositiveDefinite())
	{
		std::cout << "D vector:" << std::endl;
		const Eigen::Matrix<double, dim, 1> &vecD = ldlt.vectorD();
		std::cout << vecD << std::endl;
		const Eigen::DiagonalMatrix<double, dim> &D = vecD.asDiagonal();
		std::cout << Eigen::Matrix<double, dim, dim>(D) << std::endl;

		std::cout << "L matrix:" << std::endl;
		const Eigen::Matrix<double, dim, dim> &L = ldlt.matrixL();
		std::cout << L << std::endl;

		//
		std::cout << "reconstruct the original matrix m:" << std::endl;
		std::cout << L * D * L.transpose() << std::endl;
	}
	//else
	//	std::cout << "the matrix is not a positive definite" << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace eigen {

void cholesky()
{
	local::llt();
	std::cout << std::endl << std::endl;
	local::ldlt();
}

}  // namespace eigen
