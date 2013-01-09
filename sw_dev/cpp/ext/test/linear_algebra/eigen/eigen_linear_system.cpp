//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <Eigen/Dense>
#include <iostream>


// import most common Eigen types
//USING_PART_OF_NAMESPACE_EIGEN

namespace {
namespace local {

void linear_system_1()
{
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(100, 100);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(100, 50);

    const Eigen::MatrixXd x = A.fullPivLu().solve(b);

    const double relative_error = (A*x - b).norm() / b.norm();  // norm() is L2 norm
    std::cout << "The relative error is:\n" << relative_error << std::endl;
}

void linear_system_2()
{
    Eigen::Matrix2f A, b;
    A << 2, -1, -1, 3;
    b << 1, 2, 3, 1;
    std::cout << "Here is the matrix A:\n" << A << std::endl;
    std::cout << "Here is the right hand side b:\n" << b << std::endl;

    std::cout << "Computing LLT decomposition..." << std::endl;
    Eigen::LLT<Eigen::Matrix2f> llt;
    llt.compute(A);
    std::cout << "The solution is:\n" << llt.solve(b) << std::endl;

    A(1,1)++;
    std::cout << "The matrix A is now:\n" << A << std::endl;

    std::cout << "Computing LLT decomposition..." << std::endl;
    llt.compute(A);
    std::cout << "The solution is now:\n" << llt.solve(b) << std::endl;
}

void linear_system_3()
{
    Eigen::Matrix2f A, b;
    A << 2, -1, -1, 3;
    b << 1, 2, 3, 1;

    std::cout << "Here is the matrix A:\n" << A << std::endl;
    std::cout << "Here is the right hand side b:\n" << b << std::endl;

    const Eigen::Matrix2f x = A.ldlt().solve(b);
    std::cout << "The solution is:\n" << x << std::endl;
}

void linear_system_4()
{
    Eigen::Matrix3f A;
    Eigen::Vector3f b;

    A << 1,2,3,  4,5,6,  7,8,10;
    b << 3, 3, 4;

    std::cout << "Here is the matrix A:\n" << A << std::endl;
    std::cout << "Here is the vector b:\n" << b << std::endl;

    const Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);
    std::cout << "The solution is:\n" << x << std::endl;
}

void linear_system_5()
{
    Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 2);
    Eigen::VectorXf b = Eigen::VectorXf::Random(3);

    std::cout << "Here is the matrix A:\n" << A << std::endl;
    std::cout << "Here is the right hand side b:\n" << b << std::endl;

    std::cout << "The least-squares solution is:\n" << A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void linear_system()
{
    local::linear_system_1();
    local::linear_system_2();
    local::linear_system_3();
    local::linear_system_4();
    local::linear_system_5();
}

}  // namespace my_eigen
