//#include "stdafx.h"
#include <iostream>
#include <openblas/cblas.h>


namespace {
namespace local {

void simple_example()
{
	double A[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
	double B[6] = { 1.0, 2.0, 1.0, -3.0, 4.0, -1.0 };
	double C[9] = { .5, .5, .5, .5, .5, .5, .5, .5, .5 };
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3, 2, C, 3);

	for (int i = 0; i < 9; ++i)
		std::cout << C[i] << ", ";
	std::cout << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_openblas {

}  // namespace my_openblas

int openblas_main(int argc, char* argv[])
{
	local::simple_example();

	return 0;
}
