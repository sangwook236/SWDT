//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace eigen {

void basic_operation();
void linear_system();

void lu();
void evd();
void svd();
void qr();
void cholesky();

}  // namespace eigen

//-----------------------------------------------------------------------
// porting from Eigen2 to Eigen3
//	[ref] http://eigen.tuxfamily.org/dox/Eigen2ToEigen3.html

int eigen_main(int argc, char *argv[])
{
	//eigen::basic_operation();
	//eigen::linear_system();

	//eigen::lu();
	//eigen::evd();
	//eigen::svd();
	eigen::qr();
	//eigen::cholesky();

	return 0;
}
