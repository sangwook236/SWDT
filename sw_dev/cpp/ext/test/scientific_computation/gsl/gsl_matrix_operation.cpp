//#include "stdafx.h"
#include <gsl/gsl_blas.h>
#include <iostream>
#include <cmath>


namespace my_gsl {

void print_gsl_vector(gsl_vector *vec);
void print_gsl_vector(gsl_vector *vec, const int dim);
void print_gsl_matrix(gsl_matrix *mat);
void print_gsl_matrix(gsl_matrix *mat, const int rdim, const int cdim);

}  // namespace my_gsl

namespace {
namespace local {

void matrix_basic()
{
	// matrix view
	std::cout << ">>> matrix view" << std::endl;
	{
		const int rdim = 2, cdim = 3;
		double a[] = { 1., 2., 3., 4., 5., 6. };
		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);  // caution !!!: row-major matrix
		my_gsl::print_gsl_matrix(&A.matrix);

		gsl_matrix_set(&A.matrix, 0, 0, -13.0);
		gsl_matrix_set(&A.matrix, 1, 2, -10.0);

		//gsl_matrix_set(mat, i, j, 2.0);
		//double val = gsl_matrix_get(mat, i, j);
		//double* pval = gsl_matrix_ptr(mat, i, j);
		//const double* cpval = gsl_matrix_const_ptr(mat, i, j);

		for (int i = 0; i < rdim * cdim; ++i)
			std::cout << a[i] << ' ';
		std::cout << std::endl;
	}

	// submatrix
	std::cout << std::endl << ">>> submatrix" << std::endl;
	{
		const int rdim = 4, cdim = 4;
		const int rdim2 = 3, cdim2 = 2;
		double a[rdim * cdim] = { 0., };
		for (int i = 0; i < rdim * cdim; ++i)
			a[i] = i;

		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);
		gsl_matrix_view B = gsl_matrix_submatrix(&A.matrix, 0, 0, rdim2, cdim2);

		my_gsl::print_gsl_matrix(&B.matrix);
	}

	// row, column & diagnal of matrix
	std::cout << std::endl << ">>> row & column of matrix" << std::endl;
	{
		const int rdim = 3, cdim = 4;
		gsl_matrix* mat = gsl_matrix_alloc(rdim, cdim);

		for (int i = 0; i < rdim; ++i)
			for (int j = 0; j < cdim; ++j)
				gsl_matrix_set(mat, i, j, i * rdim + j);

		const int rowidx = 1, colidx = 2;
		gsl_vector_view row = gsl_matrix_row(mat, rowidx);
		gsl_vector_view col = gsl_matrix_column(mat, colidx);
		gsl_vector_view diag = gsl_matrix_diagonal(mat);
		//gsl_vector_view diag = gsl_matrix_subdiagonal(mat, k);
		//gsl_vector_view diag = gsl_matrix_superdiagonal(mat, k);

		//gsl_matrix_set_row(m, i, v);
		//gsl_matrix_get_col(v, m, j);
		//gsl_matrix_set_row(m, i, v);
		//gsl_matrix_get_col(v, m, j);

		my_gsl::print_gsl_vector(&row.vector);
		my_gsl::print_gsl_vector(&col.vector);
		my_gsl::print_gsl_vector(&diag.vector);

		gsl_matrix_free(mat);
	}

	// matrix copy, swap elements
	std::cout << std::endl << ">>> matrix copy, swap elements" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		const int rdim2 = 2, cdim2 = 2;
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		double b[] = { -1., -2., -3., -4. };

		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);
		gsl_matrix_view B = gsl_matrix_view_array(b, rdim2, cdim2);

		//gsl_matrix_memcpy(&A.matrix, &B.matrix);
#if defined(__GNUC__)
        gsl_matrix_view A_roi(gsl_matrix_submatrix(&A.matrix, 0, 0, rdim2, cdim2));
		gsl_matrix_memcpy(&A_roi.matrix, &B.matrix);
#else
		gsl_matrix_memcpy(&gsl_matrix_submatrix(&A.matrix, 0, 0, rdim2, cdim2).matrix, &B.matrix);
#endif

		//gsl_matrix_swap(m1, m2);

		//gsl_matrix_swap_rows(m, i, j)
		//gsl_matrix_swap_columns(m, i, j)
		//gsl_matrix_swap_rowcol(m, i, j)

		my_gsl::print_gsl_matrix(&A.matrix);
	}

	// set value, zero, identity
	std::cout << std::endl << ">>> set value, zero, identity" << std::endl;
	{
		const int dim = 3;
		gsl_matrix* mat = gsl_matrix_alloc(dim, dim);

		gsl_matrix_set_all(mat, 1.0);
		gsl_matrix_set_zero(mat);
		gsl_matrix_set_identity(mat);  // square & rectangular matrices

		gsl_matrix_free(mat);
	}

	// matrix arithmetic
	std::cout << std::endl << ">>> matrix arithmetic" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);

		//gsl_matrix_add(m1, m2);
		//gsl_matrix_sub(m1, m2);
		//gsl_matrix_mul_elements(m1, m2);
		//gsl_matrix_div_elements(m1, m2);
		gsl_matrix_scale(&A.matrix, -1.0);
		gsl_matrix_add_constant(&A.matrix, 2.0);

		my_gsl::print_gsl_matrix(&A.matrix);
	}

	// min, max of matrix
	std::cout << std::endl << ">>> min, max of vector" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);

		const double maxval = gsl_matrix_max(&A.matrix);
		const double minval = gsl_matrix_min(&A.matrix);
		std::cout << maxval << ", " << minval << std::endl;
		double maxval2, minval2;
		gsl_matrix_minmax(&A.matrix, &minval2, &maxval2);
		std::cout << maxval2 << ", " << minval2 << std::endl;

		size_t maxidx_i, maxidx_j;
		gsl_matrix_max_index(&A.matrix, &maxidx_i, &maxidx_j);
		std::cout << maxidx_i << ", " << maxidx_j << std::endl;
		size_t minidx_i, minidx_j;
		gsl_matrix_min_index(&A.matrix, &minidx_i, &minidx_j);
		std::cout << minidx_i << ", " << minidx_j << std::endl;
		size_t maxidx2_i, maxidx2_j, minidx2_i, minidx2_j;
		gsl_matrix_minmax_index(&A.matrix, &minidx2_i, &minidx2_j, &maxidx2_i, &maxidx2_j);
		std::cout << minidx2_i << ", " << minidx2_j << ", " << maxidx2_i << ", " << maxidx2_j << std::endl;
	}

	// matrix property
	std::cout << std::endl << ">>> vector property" << std::endl;
	{
		const int rdim = 3, cdim = 3;
		double a[] = { 1., 2., 3., 4., 5., 6., 7., 8., 9. };
		gsl_matrix_view A = gsl_matrix_view_array(a, rdim, cdim);

		std::cout << gsl_matrix_isnull(&A.matrix) << std::endl;
	}
}

void matrix_transpose()
{
	const int dim = 3;
	double a[] = { 11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0 };
	gsl_matrix_view A = gsl_matrix_view_array(a, dim, dim);
	gsl_matrix* AT = gsl_matrix_alloc(dim, dim);

	//gsl_matrix_transpose(&A.matrix);  // square matrix
	gsl_matrix_transpose_memcpy(AT, &A.matrix);  // square matrix (?)

	std::cout << ">>> A^T = " << std::endl;
	my_gsl::print_gsl_matrix(AT);

	gsl_matrix_free(AT);
}

/*
There are three levels of blas operations,
	Level 1: Vector operations, e.g. y = a x + y
	Level 2: Matrix-vector operations, e.g. y = a A x + b y
	Level 3: Matrix-matrix operations, e.g. C = a A B + C
*/

void matrix_vector_mulitplication()  // level 2
{
	// y = a op(A) x + b y
	// CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

	// Compute v2 = A v1
	{
		double a[] = { 1, 1, 0, 1 };

		gsl_matrix_view A = gsl_matrix_view_array(a, 2, 2);

		gsl_vector *v = gsl_vector_alloc(2);
		gsl_vector_set(v, 0, 1.0);  gsl_vector_set(v, 1, 3.0);

		// Oops !!!: not working
		// result: v = [ 0 ; 0 ]
		const bool result1 = GSL_SUCCESS == gsl_blas_dgemv(
			CblasNoTrans,
			1.0, &A.matrix, v,
			0.0, v
		);

		if (result1)
		{
			std::cout << ">>> v = A v + 0 v =>" << std::endl;
			my_gsl::print_gsl_vector(v);
		}
		else
			std::cout << ">>> error !!!" << std::endl;

		//
		gsl_vector_set(v, 0, 1.0);  gsl_vector_set(v, 1, 3.0);

		// result: v = [ 5 ; 6 ]
		const bool result2 = GSL_SUCCESS == gsl_blas_dgemv(
			CblasNoTrans,
			1.0, &A.matrix, v,
			1.0, v
		);

		if (result2)
		{
			std::cout << ">>> v = A v + 1 v =>" << std::endl;
			my_gsl::print_gsl_vector(v);
		}
		else
			std::cout << ">>> error !!!" << std::endl;

		gsl_vector_free(v);
	}

	//
	{
		double a[] = { 0.11, 0.12, 0.13, 0.21, 0.22, 0.23 };
		double b[] = { 1011, 1012, 1021 };
		double c[] = { 0.00, 0.00 };

		gsl_matrix_view A = gsl_matrix_view_array(a, 2, 3);
		gsl_vector_view x = gsl_vector_view_array(b, 3);
		gsl_vector_view y = gsl_vector_view_array(c, 2);

		const bool result = GSL_SUCCESS == gsl_blas_dgemv(
			CblasNoTrans,
			1.0, &A.matrix, &x.vector,
			0.0, &y.vector
		);

		if (result)
		{
			std::cout << ">>> y = A x =>" << std::endl;
			my_gsl::print_gsl_vector(&y.vector);
		}
		else
			std::cout << ">>> error !!!" << std::endl;
	}
}

void matrix_matrix_mulitplication()  // level 3
{
	{
		double a[] = { 0.11, 0.12, 0.13, 0.21, 0.22, 0.23 };
		double b[] = { 1011, 1012, 1021, 1022, 1031, 1032 };
		double c[] = { 0.00, 0.00, 0.00, 0.00 };

		gsl_matrix_view A = gsl_matrix_view_array(a, 2, 3);
		gsl_matrix_view B = gsl_matrix_view_array(b, 3, 2);
		gsl_matrix_view C = gsl_matrix_view_array(c, 2, 2);

		// C = a op(A) op(B) + b C
		// CBLAS_TRANSPOSE_t: CblasNoTrans, CblasTrans, CblasConjTrans

		// Compute C = A B + 0 C
		const bool result = GSL_SUCCESS == gsl_blas_dgemm(
			CblasNoTrans, CblasNoTrans,
			1.0, &A.matrix, &B.matrix,
			0.0, &C.matrix
		);

		if (result)
		{
			std::cout << ">>> C = A B =>" << std::endl;
			my_gsl::print_gsl_matrix(&C.matrix);
		}
		else
			std::cout << ">>> error !!!" << std::endl;
	}

	//
	{
		double aa[] = { 0.11, 0.12, 0.13, 0.21, 0.22, 0.23, 0.21, 0.22, 0.23 };
		double bb[] = { 1011, 1012, 1021, 1022, 1031, 1032, 1012, 1021, 1022 };
		double cc[9] = { 0.0, };

		gsl_matrix_view AA = gsl_matrix_view_array(aa, 3, 3);
		gsl_matrix_view BB = gsl_matrix_view_array(bb, 3, 3);
		gsl_matrix_view CC = gsl_matrix_view_array(cc, 3, 3);

		// Oops !!!: not working
		// Compute A = A B + 0 A
		const bool result = GSL_SUCCESS == gsl_blas_dgemm(
			CblasNoTrans, CblasNoTrans,
			1.0, &AA.matrix, &BB.matrix,
			0.0, &AA.matrix
		);

		if (result)
		{
			std::cout << ">>> A = A B =>" << std::endl;
			std::cout << ">>> Oops !!!" << std::endl;
			my_gsl::print_gsl_matrix(&AA.matrix);
		}
		else
			std::cout << ">>> error !!!" << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void matrix_operation()
{
	local::matrix_basic();
	std::cout << std::endl;
	local::matrix_transpose();
	std::cout << std::endl;
	local::matrix_vector_mulitplication();
	std::cout << std::endl;
	local::matrix_matrix_mulitplication();
}

}  // namespace my_gsl
