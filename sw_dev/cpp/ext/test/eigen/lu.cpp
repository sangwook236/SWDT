#include "stdafx.h"
#include <Eigen/LU>
#include <Eigen/Array>
#include <Eigen/Core>


void lu()
{
	//Eigen::Matrix3d m;
	//m << 1, 2, 3, 6, -1, 0, 2, 1, 1;

	typedef Eigen::Matrix<double, 5, 3> Matrix5x3;
	typedef Eigen::Matrix<double, 5, 5> Matrix5x5;

	Matrix5x3 m = Matrix5x3::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	//const Eigen::LU<Matrix5x3> lu(m);
	const Eigen::LU<Matrix5x3> lu = m.lu();
	std::cout << "LU decomposition:"
		<< std::endl << lu.matrixLU() << std::endl;

	std::cout << "L part:" << std::endl;
	Matrix5x5 l = Matrix5x5::Identity();
	l.block<5,3>(0, 0).part<Eigen::StrictlyLowerTriangular>() = lu.matrixLU();
	std::cout << l << std::endl;

	std::cout << "U part:" << std::endl;
	Matrix5x3 u = lu.matrixLU().part<Eigen::UpperTriangular>();
	std::cout << u << std::endl;

	// Oops !!! not exactly working
	std::cout << "reconstruct the original matrix m:" << std::endl;
	Matrix5x3 x = l * u;
	Matrix5x3 y;
	for (int i = 0; i < 5; ++i)
		for(int j = 0; j < 3; ++j)
			y(i, lu.permutationQ()[j]) = x(lu.permutationP()[i], j);
	std::cout << y << std::endl;  // should be equal to the original matrix m
}
