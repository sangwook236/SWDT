//#include "stdafx.h"
//#include <atlas_enum.h>
#include <atlas/clapack.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void clapack()
{
	double m[] = {
		3, 1, 3,
		1, 5, 9,
		2, 6, 5
	};
	double x[] = { -1, -1, 1 };

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			std::cout << m[i*3 + j] << " ";
		std::cout << std::endl;
	}

	int ipiv[3] = { 0, };
#if 1
	const int info = clapack_dgesv(CblasRowMajor, 3, 1, m, 3, ipiv, x, 3);
#else
    // FIXME [correct] >>
	const int info = dgesv_(CblasRowMajor, 3, 1, m, 3, ipiv, x, 3);
#endif
	if (info != 0)
		std::cerr << "failure with error: " << info << std::endl;;

	for (int i = 0; i < 3; ++i)
		std::cout << x[i] << ", " << ipiv[i] << std::endl;
}
