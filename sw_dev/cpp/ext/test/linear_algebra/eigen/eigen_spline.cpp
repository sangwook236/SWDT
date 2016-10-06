//#include "stdafx.h"
//#define EIGEN2_SUPPORT 1
#include <unsupported/Eigen/Splines>
#include <Eigen/Core>
#include <iostream>


namespace {
namespace local {

double scale_value(const double val, const double minVal, const double maxVal)
{
	//return val;  // Unscaled.
	return (val - minVal) / (maxVal - minVal);  // Scaled.
}

// REF [site] >> http://stackoverflow.com/questions/29822041/eigen-spline-interpolation-how-to-get-spline-y-value-at-arbitray-point-x
void simple_example()
{
	Eigen::VectorXd vecX(3);
	Eigen::VectorXd vecY(vecX.rows());

	vecX << 0, 15, 30;
	vecY << 0, 12, 17;

	// NOTICE [info] >> X values can be scaled down to [0, 1].
	const double minVal(vecX.minCoeff()), maxVal(vecX.maxCoeff());
	vecX = vecX.unaryExpr([minVal, maxVal](double x) { return scale_value(x, minVal, maxVal); }).transpose();

	// Define a spline type of one-dimensional points.
	typedef Eigen::Spline<double, 1> spline_type;

	// No more than cubic spline, but accept short vectors.
	const spline_type& spline = Eigen::SplineFitting<spline_type>::Interpolate(vecY.transpose(), std::min<int>(vecX.rows() - 1, 3), vecX);

	// NOTICE [warning] >> Results of scaled and unscaled values are quite different.
	std::cout << "Interpolated y by spline = " << spline(scale_value(12.34, minVal, maxVal)) << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void spline()
{
	local::simple_example();
}

}  // namespace my_eigen
