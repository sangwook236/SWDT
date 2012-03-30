//#include "stdafx.h"
#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


namespace {
namespace local {

void lu_1()
{
	//Eigen::Matrix3d m;
	//m << 1, 2, 3, 6, -1, 0, 2, 1, 1;

	typedef Eigen::Matrix<double, 5, 3> Matrix5x3;
	typedef Eigen::Matrix<double, 5, 5> Matrix5x5;

	Matrix5x3 m = Matrix5x3::Random();
	std::cout << "matrix m:" << std::endl << m << std::endl;

	//const Eigen::LU<Matrix5x3> lu(m);
	const Eigen::PartialPivLU<Matrix5x3> lu = m.lu();
	std::cout << "LU decomposition:" << std::endl << lu.matrixLU() << std::endl;

	std::cout << "L part:" << std::endl;
	Matrix5x5 l = Matrix5x5::Identity();
	l.block<5, 3>(0, 0).part<Eigen::LowerTriangular>() = lu.matrixLU();
	std::cout << l << std::endl;

	std::cout << "U part:" << std::endl;
	Matrix5x3 u = lu.matrixLU().part<Eigen::UpperTriangular>();
	std::cout << u << std::endl;

	// FIXME [correct] >> compile-time error: not exactly working
/*
	std::cout << "reconstruct the original matrix m:" << std::endl;
	Matrix5x3 x = l * u;
	Matrix5x3 y;
	for (int i = 0; i < 5; ++i)
		for(int j = 0; j < 3; ++j)
			y(i, lu.permutationQ()[j]) = x(lu.permutationP()[i], j);
	std::cout << y << std::endl;  // should be equal to the original matrix m
*/
}

void lu_2()
{
    Eigen::Matrix3f A;
    A << 1, 2, 5,  2, 1, 4,  3, 0, 3;
    std::cout << "Here is the matrix A:\n" << A << std::endl;

    Eigen::FullPivLU<Eigen::Matrix3f> lu_decomp(A);
    std::cout << "The rank of A is " << lu_decomp.rank() << std::endl;
    std::cout << "Here is a matrix whose columns form a basis of the null-space of A:\n" << lu_decomp.kernel() << std::endl;
    std::cout << "Here is a matrix whose columns form a basis of the column-space of A:\n" << lu_decomp.image(A) << std::endl; // yes, have to pass the original A
}

void lu_3()
{
    Eigen::Matrix2d A;
    A << 2, 1,  2, 0.9999999999;

    Eigen::FullPivLU<Eigen::Matrix2d> lu(A);
    std::cout << "By default, the rank of A is found to be " << lu.rank() << std::endl;
    lu.setThreshold(1e-5);
    std::cout << "With threshold 1e-5, the rank of A is found to be " << lu.rank() << std::endl;
}

}  // namespace local
}  // unnamed namespace

void lu()
{
    local::lu_1();
    local::lu_2();
    local::lu_3();
}
