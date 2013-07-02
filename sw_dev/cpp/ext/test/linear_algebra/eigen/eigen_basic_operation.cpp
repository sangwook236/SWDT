//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <iostream>


// import most common Eigen types
//USING_PART_OF_NAMESPACE_EIGEN

namespace {
namespace local {

void fixed_size_operation()
{
	Eigen::Matrix3f m3;
	m3 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
	Eigen::Matrix4f m4 = Eigen::Matrix4f::Identity();
	Eigen::Vector4i v4(1, 2, 3, 4);

	std::cout << "m3\n" << m3 << "\nm4:\n" << m4 << "\nv4:\n" << v4 << std::endl;
}

void dynamic_size_operation()
{
	for (int size = 1; size <= 4; ++size)
	{
		Eigen::MatrixXi m(size, size + 1);  // a (size)x(size+1)-matrix of int's
		for (int j = 0; j < m.cols(); ++j)   // loop over columns
			for (int i = 0; i < m.rows(); ++i)  // loop over rows
				m(i,j) = i + j * m.rows();  // to access matrix coefficients, use operator()(int,int)
		std::cout << m << "\n\n";
	}

	Eigen::VectorXf v(4);  // a vector of 4 float's to access vector coefficients, use either operator () or operator []
	v[0] = 1; v[1] = 2; v(2) = 3; v(3) = 4;
	std::cout << "\nv:\n" << v << std::endl;
}

void fixed_size_block_operation()
{
	Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
	std::cout << (4 * m).topLeftCorner<2, 2>() << std::endl;  // calls the const version

	m.topLeftCorner<2, 2>() *= 2;  // calls the non-const version
	std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
}

void dynamic_size_block_operation()
{
	Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
	std::cout << (4 * m).topLeftCorner(2, 3) << std::endl;  // calls the const version

	m.topLeftCorner(2, 3) *= 5;  // calls the non-const version
	std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
}

// define a custom template unary functor
template<typename Scalar>
struct CwiseClampOp
{
	CwiseClampOp(const Scalar &inf, const Scalar &sup)
	: m_inf(inf), m_sup(sup)
	{}

	const Scalar operator()(const Scalar &x) const { return x < m_inf ? m_inf : (x > m_sup ? m_sup : x); }

	Scalar m_inf, m_sup;
};

void coefficient_wise_unary_operation()
{
	Eigen::Matrix4d m1 = Eigen::Matrix4d::Random();
	std::cout << m1 << std::endl << "becomes: " << std::endl << m1.unaryExpr(CwiseClampOp<double>(-0.5, 0.5)) << std::endl;
}

// define a custom template binary functor
template<typename Scalar>
//struct MakeComplexOp EIGEN_EMPTY_STRUCT
struct MakeComplexOp
{
	typedef std::complex<Scalar> result_type;

	std::complex<Scalar> operator()(const Scalar &a, const Scalar &b) const
	{ return std::complex<Scalar>(a, b); }
};

void coefficient_wise_binary_operation()
{
	Eigen::Matrix4d m1 = Eigen::Matrix4d::Random(), m2 = Eigen::Matrix4d::Random();
	std::cout << m1.binaryExpr(m2, MakeComplexOp<double>()) << std::endl;
}

void matrix_arithmetic_1()
{
    //
    {
        Eigen::Matrix3f A;
        A << 1, 2, 1,  2, 1, 0,  -1, 1, 2;
        std::cout << "Here is the matrix A:\n" << A << std::endl;

        std::cout << "The determinant of A is " << A.determinant() << std::endl;
        std::cout << "The inverse of A is:\n" << A.inverse() << std::endl;
    }

    //
    {
        Eigen::VectorXf v(2);
        Eigen::MatrixXf m(2,2), n(2,2);

        v << -1,  2;
        m << 1, -2,  -3, 4;

        std::cout << "v.squaredNorm() = " << v.squaredNorm() << std::endl;
        std::cout << "v.norm() = " << v.norm() << std::endl;
        std::cout << "v.lpNorm<1>() = " << v.lpNorm<1>() << std::endl;
        std::cout << "v.lpNorm<Eigen::Infinity>() = " << v.lpNorm<Eigen::Infinity>() << std::endl;

        std::cout << std::endl;
        std::cout << "m.squaredNorm() = " << m.squaredNorm() << std::endl;
        std::cout << "m.norm() = " << m.norm() << std::endl;
        std::cout << "m.lpNorm<1>() = " << m.lpNorm<1>() << std::endl;
        std::cout << "m.lpNorm<Eigen::Infinity>() = " << m.lpNorm<Eigen::Infinity>() << std::endl;
    }
}

void matrix_arithmetic_2()
{
    //
    {
        Eigen::Matrix2d mat;
        mat << 1, 2,  3, 4;

        std::cout << "Here is mat.sum():       " << mat.sum()       << std::endl;
        std::cout << "Here is mat.prod():      " << mat.prod()      << std::endl;
        std::cout << "Here is mat.mean():      " << mat.mean()      << std::endl;
        std::cout << "Here is mat.minCoeff():  " << mat.minCoeff()  << std::endl;
        std::cout << "Here is mat.maxCoeff():  " << mat.maxCoeff()  << std::endl;
        std::cout << "Here is mat.trace():     " << mat.trace()     << std::endl;
    }

    //
    {
        Eigen::ArrayXXf a(2, 2);
        a << 1, 2,  3, 4;

        std::cout << "(a > 0).all()   = " << (a > 0).all() << std::endl;
        std::cout << "(a > 0).any()   = " << (a > 0).any() << std::endl;
        std::cout << "(a > 0).count() = " << (a > 0).count() << std::endl;
        std::cout << std::endl;
        std::cout << "(a > 2).all()   = " << (a > 2).all() << std::endl;
        std::cout << "(a > 2).any()   = " << (a > 2).any() << std::endl;
        std::cout << "(a > 2).count() = " << (a > 2).count() << std::endl;
    }

    //
    {
        Eigen::MatrixXf m(2, 2);
        m << 1, 2,  3, 4;

        //get location of maximum
        Eigen::MatrixXf::Index maxRow, maxCol;
        const float max = m.maxCoeff(&maxRow, &maxCol);

        //get location of minimum
        Eigen::MatrixXf::Index minRow, minCol;
        const float min = m.minCoeff(&minRow, &minCol);

        std::cout << "Max: " << max << ", at: " << maxRow << "," << maxCol << std::endl;
        std::cout << "Min: " << min << ", at: " << minRow << "," << minCol << std::endl;

        //
        Eigen::MatrixXf mat(2, 4);
        mat << 1, 2, 6, 9,
               3, 1, 7, 2;

        std::cout << "Column's maximum: " << std::endl << mat.colwise().maxCoeff() << std::endl;
        std::cout << "Row's maximum: " << std::endl << mat.rowwise().maxCoeff() << std::endl;

        Eigen::MatrixXf::Index maxIndex;
        const float maxNorm = mat.colwise().sum().maxCoeff(&maxIndex);

        std::cout << "Maximum sum at position " << maxIndex << std::endl;

        std::cout << "The corresponding vector is: " << std::endl;
        std::cout << mat.col(maxIndex) << std::endl;
        std::cout << "And its sum is is: " << maxNorm << std::endl;
    }

	//
	{
		Eigen::MatrixXf mat(2, 4);
		Eigen::VectorXf v1(2);
		Eigen::VectorXf v2(4);

		mat << 1, 2, 6, 9,
			3, 1, 7, 2;
		v1 << 0,
			1;
		v2 << 0, 1, 2, 3;

		//add v to each column of m
		mat.colwise() += v1;
		std::cout << "Broadcasting result: " << std::endl << mat << std::endl;

		//add v to each row of m
		//mat.rowwise() += v2;  // compile-time error
		std::cout << "Broadcasting result: " << std::endl << mat << std::endl;
	}

	//
	{
		Eigen::MatrixXf m(2, 4);
		Eigen::VectorXf v(2);

		m << 1, 23, 6, 9,
			3, 11, 7, 2;
		v << 2,
			3;

		Eigen::MatrixXf::Index index;
		// find nearest neighbour
		(m.colwise() - v).colwise().squaredNorm().minCoeff(&index);

		std::cout << "Nearest neighbour is column " << index << ":" << std::endl;
		std::cout << m.col(index) << std::endl;
	}
}

void matrix_function_operation()
{
	// sqrt
	{
		Eigen::MatrixXd m(3, 3);

		m << 101, 13, 4,
			 13, 11, 5,
			 4, 5, 37;

		//
#if 0
		const Eigen::MatrixSquareRootReturnValue<Eigen::MatrixXd> msrrv = m.sqrt();
		Eigen::MatrixXd m1;
		msrrv.evalTo(m1);
#else
		Eigen::MatrixXd m1;
		m.sqrt().evalTo(m1);
#endif
		std::cout << "matrix sqrt result: " << std::endl << m1 << std::endl;

		//
#if 0
		Eigen::MatrixSquareRoot<Eigen::MatrixXd> msr(m);
		Eigen::MatrixXd m2;
		msr.compute(m2);
#else
		Eigen::MatrixXd m2;
		Eigen::MatrixSquareRoot<Eigen::MatrixXd>(m).compute(m2);
#endif
		std::cout << "matrix sqrt result: " << std::endl << m2 << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void basic_operation()
{
    //local::fixed_size_operation();
    //local::dynamic_size_operation();

    //local::fixed_size_block_operation();
    //local::dynamic_size_block_operation();

    //local::coefficient_wise_unary_operation();
    //local::coefficient_wise_binary_operation();

    //local::matrix_arithmetic_1();
    //local::matrix_arithmetic_2();

    local::matrix_function_operation();
}

}  // namespace my_eigen
