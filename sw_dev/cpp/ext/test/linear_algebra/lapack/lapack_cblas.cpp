//#define NO_BLAS_WRAP

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <clapack/f2c.h>
//#include <clapack/blaswrap.h>
#include <clapack/clapack.h>
#else
#include <clapack.h>
#include <cblas.h>
#include <f2c.h>
#endif

#if defined(__cplusplus)
}
#endif

#if defined(abs)
#undef abs
#endif

#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_lapack {

void cblas()
{
	const double m[] = {
		3, 1, 3,
		1, 5, 9,
		2, 6, 5
	};
	const double x[] = { -1, -1, 1 };
	double y[] = { 0, 0, 0 };

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			std::cout << m[i*3 + j] << " ";
		std::cout << std::endl;
	}

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	integer rdim = 3, cdim = 3;
	double alpha = 1.0, beta = 0.0;
	integer lda = rdim;
	integer incx = 1, incy = 1;
	dgemv_((char *)"No transpose", &rdim, &cdim, &alpha, (double *)m, &lda, (double *)x, &incx, &beta, y, &incy);
#else
	cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, m, 3, x, 1, 0.0, y, 1);
#endif

	for (int i = 0; i < 3; ++i)
		std::cout << y[i] << std::endl;
}

}  // namespace my_lapack
