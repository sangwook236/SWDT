//#include "stdafx.h"
#include <cmath>
//#define EIGEN2_SUPPORT 1
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>


// Import most common Eigen types.
//USING_PART_OF_NAMESPACE_EIGEN

namespace {
namespace local {

void initialization()
{
	{
		std::cout << Eigen::Vector3d::Zero().transpose() << std::endl;
		std::cout << Eigen::VectorXd::Zero(4).transpose() << std::endl;
		std::cout << Eigen::Matrix3d::Zero() << std::endl;
		std::cout << Eigen::MatrixXd::Zero(2, 4) << std::endl;
		std::cout << Eigen::Matrix2d::Identity() << std::endl;
		//std::cout << Eigen::MatrixXd::Identity(3) << std::endl;  // Error.
		std::cout << Eigen::MatrixXd::Identity(2, 4) << std::endl;

		//
		const int size = 6;
		Eigen::MatrixXd mat1(size, size);
		mat1.topLeftCorner(size/2, size/2)     = Eigen::MatrixXd::Zero(size/2, size/2);
		mat1.topRightCorner(size/2, size/2)    = Eigen::MatrixXd::Identity(size/2, size/2);
		mat1.bottomLeftCorner(size/2, size/2)  = Eigen::MatrixXd::Identity(size/2, size/2);
		mat1.bottomRightCorner(size/2, size/2) = Eigen::MatrixXd::Zero(size/2, size/2);
		std::cout << mat1 << std::endl << std::endl;

		Eigen::MatrixXd mat2(size, size);
		mat2.topLeftCorner(size/2, size/2).setZero();
		mat2.topRightCorner(size/2, size/2).setIdentity();
		mat2.bottomLeftCorner(size/2, size/2).setIdentity();
		mat2.bottomRightCorner(size/2, size/2).setZero();
		std::cout << mat2 << std::endl;

		Eigen::MatrixXd mat3(size, size);
		mat3 << Eigen::MatrixXd::Zero(size/2, size/2), Eigen::MatrixXd::Identity(size/2, size/2),
				Eigen::MatrixXd::Identity(size/2, size/2), Eigen::MatrixXd::Zero(size/2, size/2);
		std::cout << mat3 << std::endl;

		//
	    std::cout << Eigen::Matrix3d::Constant(-3.7) << std::endl;
	    std::cout << Eigen::MatrixXd::Constant(3, 3, 1.2) << std::endl;
	    std::cout << Eigen::Matrix3d::Random() << std::endl;
		std::cout << Eigen::MatrixXd::Random(3, 4) << std::endl;
		std::cout << Eigen::ArrayXd::LinSpaced(10, 0, 90).transpose() << std::endl;
		//std::cout << Eigen::ArrayXXd::LinSpaced(10, 0, 90) << std::endl;
	}

	// Comma initializer.
	{
		Eigen::Matrix3f m1;
		m1 << 11, 12, 13,  14, 15, 16,  17, 18, 19;  // Row-major.

		std::cout << "Comma initialized matrix = " << std::endl << m1 << std::endl;

		Eigen::MatrixXf m2(2, 4);
		m2 << 21, 22, 23, 24,  25, 26, 27, 28;  // Row-major.

		std::cout << "Comma initialized matrix = " << std::endl << m2 << std::endl;

		Eigen::MatrixXf m3(4, 2);
		m3 << 31, 32,  33, 34,  35, 36,  37, 38;  // Row-major.

		std::cout << "Comma initialized matrix = " << std::endl << m3 << std::endl;
	}

	{
		Eigen::Matrix3f m;
		m.row(0) << 1, 2, 3;
		m.block(1, 0, 2, 2) << 4, 5, 7, 8;
		m.col(2).tail(2) << 6, 9;

		std::cout << m << std::endl;
	}

	// Use temporary objects.
	{
		Eigen::MatrixXf mat = Eigen::MatrixXf::Random(2, 3);
		std::cout << mat << std::endl;

		// The finished() method is necessary here to get the actual matrix object once the comma initialization of our temporary submatrix is done.
		mat = (Eigen::MatrixXf(2, 2) << 0, 1, 1, 0).finished() * mat;
		std::cout << mat << std::endl;
	}
}

void concatenation()
{
	Eigen::MatrixXd A(2, 4);
	Eigen::MatrixXd B(2, 4);

	A << 19, 18, 17, 16,  15, 14, 13, 12;
	B << 29, 28, 27, 26,  25, 24, 23, 22;

	Eigen::MatrixXd C(A.rows(), A.cols() + B.cols());
	C << A, B;  // Horizontally concatenated.

	C(0, 0) = 1; C(0, 1) = 2; C(0, 2) = 3; C(0, 3) = 4; C(0, 4) = 5; C(0, 5) = 6; C(0, 6) = 7; C(0, 7) = 8;

	std::cout << "Horizontally concatenated matrix = " << std::endl << C << std::endl;

	Eigen::MatrixXd D(A.rows() + B.rows(), A.cols());
	D << A, B;  // Vertically concatenated.

	std::cout << "Vertically concatenated matrix = " << std::endl << D << std::endl;
}

void matrix_or_vector_expression_mapping()
{
	{
		int arr[] = { 1, 2, 3, 4, 5, 6, 7, 8, };
		std::cout << "Column-major = " << std::endl << Eigen::Map<Eigen::Matrix<int, 2, 4> >(arr) << std::endl;
		std::cout << "Row-major = " << std::endl << Eigen::Map<Eigen::Matrix<int, 2, 4, Eigen::RowMajor> >(arr) << std::endl;
		std::cout << "Row-major using stride = " << std::endl << Eigen::Map<Eigen::Matrix<int, 2, 4>, Eigen::Unaligned, Eigen::Stride<1, 4> >(arr) << std::endl;

		//
		int arr2[15];
		for (int i = 0; i < 15; ++i) arr2[i] = i;

		// Use an inner stride, the pointer increment between two consecutive coefficients.
		std::cout << "Matrix using inner stride = " << std::endl << Eigen::Map<Eigen::VectorXi, 0, Eigen::InnerStride<2> >(arr2, 6) << std::endl;

		// Use an outer stride, the pointer increment between two consecutive columns.
		std::cout << "Matrix using outer stride = " << std::endl << Eigen::Map<Eigen::MatrixXi, 0, Eigen::OuterStride<> >(arr2, 3, 3, Eigen::OuterStride<>(5)) << std::endl;
	}

	//
	{
		double *p = new double [15];
		for (int i = 0; i < 15; ++i) p[i] = i + 100;

		Eigen::Map<Eigen::MatrixXd> P(p, 2, 6);  // Column-major. P is a wrapper of p.
		//Eigen::MatrixXd P = Eigen::Map<Eigen::MatrixXd>(p, 2, 6);  // P is a copied object of p.

		delete [] p;
		p = NULL;

		std::cout << "Matrix after deletion = " << std::endl << P << std::endl;
	}

	//
	{
		double x[] = { 19, 18, 17,  16, 15, 14,  13, 12, 11, };
		Eigen::Map<Eigen::Matrix3d> X(x);  // Column-major. X is a wrapper of x.
		//Eigen::Matrix3d X(Eigen::Map<Eigen::Matrix3d>(x));  // X is a copied object of x.

		X(0, 0) = -10;
		X(1, 0) = -20;

		// NOTE [info] >> Change in X affects x directly.
		std::cout << "X = " << std::endl << X << std::endl;
		std::cout << "x = ";
		for (int i = 0; i < 9; ++i)
			std::cout << x[i] << ", ";
		std::cout << std::endl;

		//
		const double *pX = X.data();  // Column-major.
		std::cout << "X.data() = ";
		for (int i = 0; i < 9; ++i)
			std::cout << pX[i] << ", ";
		std::cout << std::endl;
	}

	//
	{
		double a[] = { 29, 28, 27, 26,  25, 24, 23, 22 };
		double b[] = { 39, 38, 37, 36,  35, 34, 33, 32 };

		{
			Eigen::Map<Eigen::MatrixXd> A(a, 2, 4);
			Eigen::Map<Eigen::MatrixXd> B(b, 2, 4);

			Eigen::MatrixXd C(A.rows(), A.cols() + B.cols());
			C << A, B;  // Horizontally concatenated.

			std::cout << "C = " << std::endl << C << std::endl;

			C(0, 0) = -1; C(0, 1) = -2; C(0, 2) = -3; C(0, 3) = -4; C(0, 4) = -5; C(0, 5) = -6; C(0, 6) = -7; C(0, 7) = -8;

			A = C.topLeftCorner(A.rows(), A.cols());
			B = C.topRightCorner(B.rows(), B.cols());

			std::cout << "A = " << std::endl << A << std::endl;
			std::cout << "B = " << std::endl << B << std::endl;
		}

		// NOTE [info] >> Change in A & B affects a & b directly.
		std::cout << "a = ";
		for (int i = 0; i < 8; ++i)
			std::cout << a[i] << ", ";
		std::cout << std::endl;
		std::cout << "b = ";
		for (int i = 0; i < 8; ++i)
			std::cout << b[i] << ", ";
		std::cout << std::endl;
	}
}

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
		Eigen::MatrixXi m(size, size + 1);  // A (size)x(size+1)-matrix of int's.
		for (int j = 0; j < m.cols(); ++j)   // Loop over columns.
			for (int i = 0; i < m.rows(); ++i)  // Loop over rows.
				m(i,j) = i + j * m.rows();  // To access matrix coefficients, use operator()(int,int).
		std::cout << m << "\n\n";
	}

	Eigen::VectorXf v(4);  // A vector of 4 float's to access vector coefficients, use either operator () or operator [].
	v[0] = 1; v[1] = 2; v(2) = 3; v(3) = 4;
	std::cout << "\nv:\n" << v << std::endl;
}

void fixed_size_block_operation()
{
	Eigen::Matrix3d m = Eigen::Matrix3d::Identity();
	std::cout << (4 * m).topLeftCorner<2, 2>() << std::endl;  // Calls the const version.

	m.topLeftCorner<2, 2>() *= 2;  // Calls the non-const version.
	std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
}

void dynamic_size_block_operation()
{
	Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
	std::cout << (4 * m).topLeftCorner(2, 3) << std::endl;  // Calls the const version.

	m.topLeftCorner(2, 3) *= 5;  // Calls the non-const version.
	std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
}

// Define a custom template unary functor.
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

// Define a custom template binary functor.
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
	// Sub-matrix.
	{
		Eigen::Matrix4d m = Eigen::Matrix4d::Random();
		std::cout << "Here is the matrix m:" << std::endl << m << std::endl;
		// col(), row(), leftCols(), middleCols(), rightCols(), topRows(), middleRows(), bottomRows(), topLeftCorner(), topRightCorner(), bottomLeftCorner(), bottomRightCorner().
		// block(), head(), tail(), segment().
		// colwise(), rowwise().
		std::cout << "Here is m.bottomRightCorner(2, 2):" << std::endl;
		std::cout << m.bottomRightCorner(2, 2) << std::endl;
		m.bottomRightCorner(2, 2).setZero();
		std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
	}
    //
    {
        Eigen::Matrix3f A;
        A << 1, 2, 1,  2, 1, 0,  -1, 1, 2;
        std::cout << "Here is the matrix A:\n" << A << std::endl;
        std::cout << "The transpose of A is:\n" << A.transpose() << std::endl;
		std::cout << "The conjugate of A is:\n" << A.conjugate() << std::endl;
		std::cout << "The adjoint (conjugate transpose) of A is:\n" << A.adjoint() << std::endl;
        std::cout << "The determinant of A is " << A.determinant() << std::endl;
        std::cout << "The inverse of A is:\n" << A.inverse() << std::endl;
    }
    //
    {
        Eigen::VectorXf v(2);
        Eigen::MatrixXf m(2, 2), n(2, 2);
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
	// Coefficient-wise operation.
	{
		Eigen::Matrix3f mat;
		Eigen::Vector4f v1, v2;
		mat << 1, 2, 6,
			   -9, 3, -1,
			   7, 2, -2;
		v1 << 5, 2, 1, 7;
		v2 << 3, 1, 2, 4;
		std::cout << "mat.cwiseAbs()     =\n" << mat.cwiseAbs() << std::endl;
		std::cout << "mat.cwiseAbs2()    =\n" << mat.cwiseAbs2() << std::endl;
		std::cout << "mat.cwiseInverse() =\n" << mat.cwiseInverse() << std::endl;
		std::cout << "mat.cwiseSqrt()    =\n" << mat.cwiseSqrt() << std::endl;
		std::cout << "mat.cwiseAbs()     =\n" << mat.cwiseAbs() << std::endl;
		const int count1 = mat.cwiseEqual(Eigen::Matrix3f::Identity()).count();
		std::cout << "the number of mat.cwiseEqual(Eigen::Matrix3f::Identity())    = " << count1 << std::endl;
		const int count2 = mat.cwiseNotEqual(Eigen::Matrix3f::Identity()).count();
		std::cout << "the number of mat.cwiseNotEqual(Eigen::Matrix3f::Identity()) = " << count2 << std::endl;
		std::cout << "v1.cwiseMax(3)  =\n" << v1.cwiseMax(3) << std::endl;
		std::cout << "v1.cwiseMax(v2) =\n" << v1.cwiseMax(v2) << std::endl;
		std::cout << "v1.cwiseMin(3)  =\n" << v1.cwiseMin(3) << std::endl;
		std::cout << "v1.cwiseMin(v2) =\n" << v1.cwiseMin(v2) << std::endl;
		std::cout << "v1.cwiseQuotient(v2) =\n" << v1.cwiseQuotient(v2) << std::endl;
		std::cout << "v1.cwiseProduct(v2) =\n" << v1.cwiseProduct(v2) << std::endl;
	}
	//
	{
		Eigen::Vector3f v1;
		Eigen::Vector3f v2;
		v1 << 1, 2, 3;
		v2 << 6, 5, 4;
		std::cout << "dot(v1, v2) = " << v1.dot(v2) << std::endl;
		std::cout << "cross(v1, v2) = " << v1.cross(v2) << std::endl;
		std::cout << "outer(v1, v2) = " << v1 * v2.transpose() << std::endl;
	}
    //
    {
        Eigen::Matrix2d mat;
        mat << 1, 2,  3, 4;
        std::cout << "Here is mat.sum():       " << mat.sum() << std::endl;
        std::cout << "Here is mat.prod():      " << mat.prod() << std::endl;
        std::cout << "Here is mat.mean():      " << mat.mean() << std::endl;
        std::cout << "Here is mat.minCoeff():  " << mat.minCoeff() << std::endl;
        std::cout << "Here is mat.maxCoeff():  " << mat.maxCoeff() << std::endl;
        std::cout << "Here is mat.trace():     " << mat.trace() << std::endl;
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
        // Get location of maximum.
        Eigen::MatrixXf::Index maxRow, maxCol;
        const float max = m.maxCoeff(&maxRow, &maxCol);
        // Get location of minimum.
        Eigen::MatrixXf::Index minRow, minCol;
        const float min = m.minCoeff(&minRow, &minCol);
        std::cout << "Max: " << max << ", at: " << maxRow << "," << maxCol << std::endl;
        std::cout << "Min: " << min << ", at: " << minRow << "," << minCol << std::endl;
	}
	// Column-wise, row-wise.
    {
		//
        Eigen::MatrixXf mat1(2, 4);
        mat1 << 1, 2, 6, 9,
                3, 1, 7, 2;
        std::cout << "Column's maximum: " << std::endl << mat1.colwise().maxCoeff() << std::endl;
        std::cout << "Row's maximum: " << std::endl << mat1.rowwise().maxCoeff() << std::endl;
        Eigen::MatrixXf::Index maxIndex;
        const float maxNorm = mat1.colwise().sum().maxCoeff(&maxIndex);
        std::cout << "Maximum sum at position " << maxIndex << std::endl;
        std::cout << "The corresponding vector is: " << std::endl;
        std::cout << mat1.col(maxIndex) << std::endl;
        std::cout << "And its sum is: " << maxNorm << std::endl;
		//
		Eigen::MatrixXf mat2(2, 4);
		Eigen::VectorXf v1(2);
		Eigen::VectorXf v2(4);
		mat2 << 1, 2, 6, 9,
			    3, 1, 7, 2;
		v1 << 2, 3;
		v2 << 5, 4, 3, 2;
		// Add v to each column of m.
		mat2.colwise() += v1;
		std::cout << "Broadcasting result: " << std::endl << mat2 << std::endl;
		// Add v to each row of m.
		//mat2.rowwise() += v2;  // Compile-time error.
		mat2.rowwise() += v2.transpose();
		std::cout << "Broadcasting result: " << std::endl << mat2 << std::endl;
		//
		Eigen::MatrixXf mat3(2, 4);
		mat3 << 1, 2, 6, 9,
			    3, 1, 7, 2;
		//std::cout << "mat3.colwise() / v1 = \n" << mat3.colwise() / v1 << std::endl;  // Compile-time error.
		//std::cout << "mat3.rowwise() / v2 = \n" << mat3.rowwise() / v2.transpose() << std::endl;  // Compile-time error.
		const Eigen::VectorXf normvec = mat3.colwise().norm();
		for (int c = 0; c < mat3.cols(); ++c)
			mat3.col(c) /= normvec(c);
		std::cout << "(normalized) mat3 =\n" << mat3 << std::endl;
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
		// Find nearest neighbour.
		(m.colwise() - v).colwise().squaredNorm().minCoeff(&index);
		std::cout << "Nearest neighbour is column " << index << ":" << std::endl;
		std::cout << m.col(index) << std::endl;
	}
}

void matrix_function_operation()
{
	// Sqrt.
	{
		Eigen::MatrixXd m(3, 3);

		m << 101, 13, 4,
			 13, 11, 5,
			 4, 5, 37;
		std::cout << m << std::endl;

		//
#if 0
		const Eigen::MatrixSquareRootReturnValue<Eigen::MatrixXd> msrrv = m.sqrt();
		Eigen::MatrixXd m1;
		msrrv.evalTo(m1);
#else
		Eigen::MatrixXd m1;
		m.sqrt().evalTo(m1);
#endif
		std::cout << "Matrix sqrt result: " << std::endl << m1 << std::endl;

		//
#if 0
		Eigen::MatrixSquareRoot<Eigen::MatrixXd> msr(m);
		Eigen::MatrixXd m2;
		msr.compute(m2);
#elif 0
		Eigen::MatrixXd m2;
		Eigen::MatrixSquareRoot<Eigen::MatrixXd>(m).compute(m2);
#else
		const Eigen::MatrixXd m2(m.sqrt());
#endif
		std::cout << "Matrix sqrt result: " << std::endl << m2 << std::endl;
	}
}

// REF [site] >> https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html
void transformation()
{
	// Rotation.

	{
		const double angle = M_PI / 3.0;
		//Eigen::Rotation2D<double> r(angle);
		Eigen::Rotation2Dd r(angle);

		std::cout << "r.angle() = " << r.angle() << std::endl;
		std::cout << "r.smallestPositiveAngle() = " << r.smallestPositiveAngle() << std::endl;
		std::cout << "r.smallestAngle() = " << r.smallestAngle() << std::endl;

		std::cout << "r.inverse():\n" << r.inverse().toRotationMatrix() << std::endl;

		//r = Eigen::Rotation2Dd::Identity();
		r.fromRotationMatrix(Eigen::Rotation2Dd::Matrix2::Identity());
		std::cout << r.toRotationMatrix() << std::endl;

		// Spherical interpolation.
		//r.slerp();
	}

	{
		const double angle = M_PI / 3.0;
		const Eigen::Vector3d axis(0, 0, 1);

		//Eigen::Quaternion<double> q(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5));
		Eigen::Quaterniond q(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5));
		//Eigen::Quaterniond q(Eigen::Quaterniond::AngleAxisType(angle, axis));

		std::cout << "q.x() = " << q.x() << std::endl;
		std::cout << "q.y() = " << q.y() << std::endl;
		std::cout << "q.z() = " << q.z() << std::endl;
		std::cout << "q.w() = " << q.w() << std::endl;
		std::cout << "q.vec() = " << q.vec().transpose() << std::endl;
		std::cout << "q.coeffs() = " << q.coeffs().transpose() << std::endl;

		std::cout << "q.norm() = " << q.norm() << std::endl;
		std::cout << "q.squaredNorm() = " << q.squaredNorm() << std::endl;

		std::cout << "q.inverse() = " << q.inverse() << std::endl;
		std::cout << "q.conjugate() = " << q.conjugate() << std::endl;

		std::cout << "q.toRotationMatrix():\n" << q.toRotationMatrix() << std::endl;

		q = Eigen::Quaterniond::Identity();
		std::cout << q << std::endl;
	}

	{
		const double angle = M_PI / 3.0;
		const Eigen::Vector3d axis(0, 0, 1);

		//Eigen::AngleAxis<double> a(angle, axis);
		Eigen::AngleAxisd a(angle, axis);
		//Eigen::AngleAxisd a(Eigen::Quaterniond(std::cos(angle * 0.5), 0.0, 0.0, std::sin(angle * 0.5)));

		std::cout << "a.angle() = " << a.angle() << std::endl;
		std::cout << "a.axis() = " << a.axis().transpose() << std::endl;

		std::cout << "a.inverse():\n" << a.inverse().toRotationMatrix() << std::endl;

		//a = Eigen::AngleAxisd::Identity();
		a.fromRotationMatrix(Eigen::AngleAxisd::Matrix3::Identity());
		std::cout << a.toRotationMatrix() << std::endl;
	}

	//-----
	// Scaling.

	{
		//Eigen::UniformScaling<double> s(2.0);
		Eigen::UniformScaling<double> s = Eigen::Scaling(2.0);

		std::cout << "s.factor() = " << s.factor() << std::endl;
		std::cout << "s.inverse() = " << s.inverse().factor() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 2> s(2.0, 4.0);
		Eigen::DiagonalMatrix<double, 2> s = Eigen::Scaling(2.0, 4.0);

		std::cout << "s.diagonal() = " << s.diagonal().transpose() << std::endl;
		std::cout << "s.inverse() = " << s.inverse().diagonal().transpose() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 3> s(2.0, 4.0, 8.0);
		Eigen::DiagonalMatrix<double, 3> s = Eigen::Scaling(2.0, 4.0, 8.0);

		std::cout << "s.diagonal() = " << s.diagonal().transpose() << std::endl;
		std::cout << "s.inverse() = " << s.inverse().diagonal().transpose() << std::endl;
	}

	{
		//Eigen::DiagonalMatrix<double, 5> s(2.0, 4.0, 8.0, 4.0, 2.0);
		Eigen::DiagonalMatrix<double, 5> s = Eigen::Scaling(Eigen::Vector<double, 5>(2.0, 4.0, 8.0, 4.0, 2.0));

		std::cout << "s.diagonal() = " << s.diagonal().transpose() << std::endl;
		std::cout << "s.inverse() = " << s.inverse().diagonal().transpose() << std::endl;
	}

	//-----
	// Translation.

	{
		//Eigen::Translation<double, 2> t(1.0, 2.0);
		Eigen::Translation2d t(1.0, 2.0);

		std::cout << "t.x() = " << t.x() << std::endl;
		std::cout << "t.y() = " << t.y() << std::endl;
		std::cout << "t.vector() = " << t.vector().transpose() << std::endl;
		std::cout << "t.translation() = " << t.translation().transpose() << std::endl;

		std::cout << "t.inverse() = " << t.inverse().vector().transpose() << std::endl;

		t = Eigen::Translation2d::Identity();
		std::cout << t.vector().transpose() << std::endl;
	}

	{
		//Eigen::Translation<double, 3> t(1.0, 2.0, 3.0);
		Eigen::Translation3d t(1.0, 2.0, 3.0);

		std::cout << "t.x() = " << t.x() << std::endl;
		std::cout << "t.y() = " << t.y() << std::endl;
		std::cout << "t.z() = " << t.z() << std::endl;
		std::cout << "t.vector() = " << t.vector().transpose() << std::endl;
		std::cout << "t.translation() = " << t.translation().transpose() << std::endl;

		std::cout << "t.inverse() = " << t.inverse().vector().transpose() << std::endl;

		t = Eigen::Translation3d::Identity();
		std::cout << t.vector().transpose() << std::endl;
	}

	{
		Eigen::Translation<double, 5> t(Eigen::Vector<double, 5>(1.0, 2.0, 3.0, 4.0, 5.0));

		std::cout << "t.x() = " << t.x() << std::endl;
		std::cout << "t.y() = " << t.y() << std::endl;
		std::cout << "t.z() = " << t.z() << std::endl;
		std::cout << "t.vector() = " << t.vector().transpose() << std::endl;
		std::cout << "t.translation() = " << t.translation().transpose() << std::endl;

		std::cout << "t.inverse() = " << t.inverse().vector().transpose() << std::endl;

		t = Eigen::Translation<double, 5>::Identity();
		std::cout << t.vector().transpose() << std::endl;
	}

	//-----
	// Affine transformation.

	{
		const Eigen::Vector3d translation(1, 2, 3);
		const double angle(M_PI / 3.0);
		const Eigen::Vector3d axis(0, 0, 1);
		const double scale = 2.0;

		//Eigen::Transform<double, 3, Eigen::Isometry> t = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		Eigen::Transform<double, 3, Eigen::Affine> t = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::Transform<double, 3, Eigen::AffineCompact> t = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);
		//Eigen::Transform<double, 3, Eigen::Projective> t = Eigen::Translation3d(translation) * Eigen::AngleAxisd(angle, axis) * Eigen::Scaling(scale);

		std::cout << "t.matrix():\n" << t.matrix() << std::endl;
		std::cout << "t.linear():\n" << t.linear() << std::endl;
		std::cout << "t.affine():\n" << t.affine() << std::endl;
		std::cout << "t.translation() = " << t.translation().transpose() << std::endl;
		std::cout << "t.rotation():\n" << t.rotation() << std::endl;
		std::cout << "t.inverse():\n" << t.inverse().matrix() << std::endl;

		t.makeAffine();
		std::cout << "t.makeAffine():\n" << t.matrix() << std::endl;

		t = Eigen::Transform<double, 3, Eigen::Affine>::Identity();
		std::cout << "t.matrix():\n" << t.matrix() << std::endl;
	}

	//-----
	// Linear transformation.

	{
		const float angle(M_PIf / 3.0f);
		const Eigen::Vector3f axis(0, 0, 1);
		const float scale = 2.0f;

		//Eigen::Matrix<float, 2, 2> m = Eigen::Rotation2Df(angle) * Eigen::Scaling(scale);
		//Eigen::Matrix2f m = Eigen::Rotation2Df(angle) * Eigen::Scaling(scale);
		//Eigen::Matrix<float, 3, 3> m = Eigen::AngleAxisf(angle, axis) * Eigen::Scaling(scale);
		Eigen::Matrix3f m = Eigen::AngleAxisf(angle, axis) * Eigen::Scaling(scale);

		std::cout << "m:\n" << m << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void basic_operation()
{
	//local::initialization();
	//local::concatenation();
	//local::matrix_or_vector_expression_mapping();

	//local::fixed_size_operation();
	//local::dynamic_size_operation();
	//local::fixed_size_block_operation();
	//local::dynamic_size_block_operation();
    
	//local::coefficient_wise_unary_operation();
	//local::coefficient_wise_binary_operation();

	//local::matrix_arithmetic_1();
	//local::matrix_arithmetic_2();
	//local::matrix_function_operation();

	local::transformation();
}

}  // namespace my_eigen
