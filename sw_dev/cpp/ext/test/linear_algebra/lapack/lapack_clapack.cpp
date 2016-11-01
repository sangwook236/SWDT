//#define NO_BLAS_WRAP
#include <cmath>

#if defined(__cplusplus)
extern "C" {
#endif

#if defined(WIN32) || defined(_WIN32)
#include <clapack/f2c.h>
//#include <clapack/blaswrap.h>
#include <clapack/clapack.h>
#else
#include <f2c.h>
#include <cblas.h>
#include <clapack.h>
#endif

#if defined(__cplusplus)
}
#endif

#if defined(abs)
#undef abs
#endif
#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif

#include <iostream>
#include <cstring>
#include <cassert>


namespace {
namespace local {

void print_matrix(const char *msg, const real *mat, const integer row, const integer col)
{
	std::cout << msg << std::endl;
	for (int i = 0; i < row; ++i)
	{
		std::cout << '\t';
		for (int j = 0; j < col; ++j)
			std::cout << mat[i * col + j] << ", ";
		std::cout << std::endl;
	}
}

bool multiply_matrix(const real *mat1, const integer row1, const integer col1, const real *mat2, const integer row2, const integer col2, real *mat)
{
	if (col1 != row2)
	{
		mat = 0L;
		return false;
	}

	memset(mat, 0, sizeof(real) * row1 * col2);
	for (int i = 0; i < row1; ++i)
		for (int j = 0; j < col2; ++j)
			for (int k = 0; k < col1; ++k)
			{
				mat[i * col2 + j] += mat1[i * col1 + k] * mat2[k * col2 + j];
			}

	return true;
}

void transpose_matrix(const real *mat1, const integer row1, const integer col1, real *mat)
{
	for (int i = 0; i < row1; ++i)
		for (int j = 0; j < col1; ++j)
			mat[j * row1 + i] = mat1[i * col1 + j];
}

}  // namespace local
}  // unnamed namespace

namespace my_lapack {

#define __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM 0
void clapack()
{
	{
		integer dim = 2;

		real *mat = new real [dim * dim];
		mat[0] = 1.0; mat[1] = 4.0;
		mat[2] = 4.0; mat[3] = 1.0;
		local::print_matrix("mat:", mat, dim, dim);

		real *eigval = new real [dim];
		integer lwork = (3 * dim - 1) * 2;
		integer info;

		real *work = new real [lwork];

#if 1
		ssyev_("V", "U", &dim, mat, &dim, eigval, work, &lwork, &info);
#else
		clapck_ssyev("V", "U", &dim, mat, &dim, eigval, work, &lwork, &info);
#endif
		local::print_matrix("eigenvalue:", eigval, 1, dim);
		local::print_matrix("eigenvector:", mat, dim, dim);

		delete [] mat;
		delete [] eigval;
		delete [] work;
	}

	{
		integer row_dim = 2, col_dim = 3;

		real *mat = new real [row_dim * col_dim];
		mat[0] = 1.0; mat[1] = 4.0; mat[2] = 2.0;
		mat[3] = 9.0; mat[4] = 1.0; mat[5] = 1.0;
		local::print_matrix("mat:", mat, row_dim, col_dim);

		real *mat_T = new real [row_dim * col_dim];
		local::transpose_matrix(mat, row_dim, col_dim, mat_T);
		local::print_matrix("mat^T:", mat_T, col_dim, row_dim);

		real *mat2 = new real [row_dim * row_dim];
		local::multiply_matrix(mat, row_dim, col_dim, mat_T, col_dim, row_dim, mat2);
		local::print_matrix("mat^T:", mat2, row_dim, row_dim);

		delete [] mat;
		delete [] mat_T;

		real *eigval = new real [row_dim];
		integer info;
#if defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 1
		integer lwork = (1 + 6*row_dim + 2*row_dim*row_dim) * 2;
		integer liwork = (3 + 5*row_dim) * 2;

		integer *iwork = new integer [liwork];
#elif defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 2
		real bound = 0.0f;
		integer index = 1;
		real abs_tol = 0.0f;
		integer neigval = 0;
		integer lwork = (26 * row_dim) * 2;
		integer liwork = (10 * row_dim) * 2;

		real *eigvec = new real [row_dim * row_dim];
		integer *isuppz = new integer [row_dim * 2];
		integer *iwork = new integer [liwork];
#else
		integer lwork = (3 * row_dim - 1) * 2;
#endif

		real *work = new real [lwork];

#if defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 1
		ssyevd_("V", "U", &row_dim, mat2, &row_dim, eigval, work, &lwork, iwork, &liwork, &info);
#elif defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 2
		ssyevr_("V", "A", "U", &row_dim, mat2, &row_dim, &bound, &bound, &index, &index, &abs_tol, &neigval, eigval, eigvec, &row_dim, isuppz, work, &lwork, iwork, &liwork, &info);
#else
		ssyev_("V", "U", &row_dim, mat2, &row_dim, eigval, work, &lwork, &info);
#endif
		assert(0 == info);

		local::print_matrix("eigenvalue:", eigval, 1, row_dim);
#if defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 1
		local::print_matrix("eigenvector:", mat2, row_dim, row_dim);
#elif defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 2
		local::print_matrix("eigenvector:", eigvec, row_dim, row_dim);
#else
		local::print_matrix("eigenvector:", mat2, row_dim, row_dim);
#endif

		delete [] mat2;
		delete [] eigval;
		delete [] work;
#if defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 1
		delete [] iwork;
#elif defined(__CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM) && __CLAPACK_DRIVER_TYPE_FOR_EIGENPROBLEM == 2
		delete [] eigvec;
		delete [] isuppz;
		delete [] iwork;
#endif
	}
}

}  // namespace my_lapack
