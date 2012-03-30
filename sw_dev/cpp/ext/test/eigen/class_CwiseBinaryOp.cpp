//#include "stdafx.h"
#include <Eigen/Core>
#include <Eigen/Array>


USING_PART_OF_NAMESPACE_EIGEN

// define a custom template binary functor
template<typename Scalar> struct MakeComplexOp EIGEN_EMPTY_STRUCT
{
	typedef std::complex<Scalar> result_type;

	std::complex<Scalar> operator()(const Scalar &a, const Scalar &b) const
	{ return std::complex<Scalar>(a, b); }
};

int coefficient_wise_biary_operator(int, char**)
{
	Matrix4d m1 = Matrix4d::Random(), m2 = Matrix4d::Random();
	std::cout << m1.binaryExpr(m2, MakeComplexOp<double>()) << std::endl;

	return 0;
}
