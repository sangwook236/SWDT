//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_eigen {

void basic_operation();
void linear_system();

void lu();
void evd();
void svd();
void qr();
void cholesky();

}  // namespace my_eigen

//-----------------------------------------------------------------------
// porting from Eigen2 to Eigen3
//	[ref] http://eigen.tuxfamily.org/dox/Eigen2ToEigen3.html

int eigen_main(int argc, char *argv[])
{
	my_eigen::basic_operation();
	//my_eigen::linear_system();

	//my_eigen::lu();
	//my_eigen::evd();
	//my_eigen::svd();
	//my_eigen::qr();
	//my_eigen::cholesky();

	return 0;
}
