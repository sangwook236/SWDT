//#include "stdafx.h"
#if !defined(__FUNCTION__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __FUNCTION__ L""
//#else
#define __FUNCTION__ ""
//#endif
#endif
#if !defined(__func__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __func__ L""
//#else
#define __func__ ""
//#endif
#endif

#include <scythestat/matrix.h>
#include <scythestat/la.h>
#include <scythestat/ide.h>
#include <algorithm>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_scythe {

// [ref] "The Scythe Statistical Library: An Open Source C++ Library for Statistical Computation", Daniel Pemstein, Kevin M. Quinn, and Andrew D. Martin, JSS 2011.
void matrix_operation()
{
	{
		const double vals[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
		//scythe::Matrix<double, scythe::Col, scythe::Concrete> Xcol(3, 4, vals);
		scythe::Matrix<> Xcol(3, 4, vals);

		scythe::Matrix<double, scythe::Row, scythe::View> Xrow(3, 4, false);
		Xrow = 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12;

		std::cout << "Xcol = " << Xcol << std::endl << Xrow << std::endl;

		Xcol(0) = Xcol(3) = Xcol(6) = Xcol(9) = 0;
		std::cout << "Xcol = " << Xcol << std::endl;

		// sub-matrix.
		scythe::Matrix<> X(3, 4, false);
		X = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;

		scythe::Matrix<double, scythe::Col, scythe::View> a = X(scythe::_, 0);
		scythe::Matrix<double, scythe::Col, scythe::View> b = X(1, scythe::_);
		scythe::Matrix<double, scythe::Col, scythe::View> C = X(0, 2, 1, 3);

		std::cout << "a = " << a << std::endl << "b = " << b << std::endl << "C = " << C << std::endl;
	}

	{
		scythe::Matrix<> X(3, 4, false);
		X = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
		scythe::Matrix<> y(3, 1, false);
		y = -1, 0, 1;

		std::random_shuffle(X.begin(), X.end());
		std::cout << "X_shuffled = " << X << std::endl;
		std::sort(X(scythe::_, 1).begin(), X(scythe::_, 1).end());
		std::cout << "X_sorted = " << X << std::endl;

		//
		X = 7, 11, 8, 1, 4, 6, 5, 2, 12, 10, 3, 9;

		// invpd() : the inverse of a positive definite symmetric matrix.
		// crossprod(X) : X^T * X.
		// t(X) : the transpose of a matrix, X^T.
		scythe::Matrix<> beta_hat = scythe::invpd(scythe::crossprod(X)) * scythe::t(X) * y;
		std::cout << "beta_hat = " << beta_hat << std::endl;
	}
}

}  // namespace my_scythe
