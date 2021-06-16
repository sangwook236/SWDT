//#include "stdafx.h"
#include <gsl/gsl_linalg.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void print_gsl_vector(gsl_vector *vec, const int dim);
void print_gsl_matrix(gsl_matrix *mat, const int rdim, const int cdim);

void lu()
{
	const int dim = 4;
	double a_data[] = {
		0.18, 0.60, 0.57, 0.96,
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.14, 0.30, 0.97, 0.66,  // for a singular case.
		//0.51, 0.13, 0.19, 0.85
	};
	gsl_matrix_view m = gsl_matrix_view_array(a_data, dim, dim);
	gsl_permutation *p = gsl_permutation_alloc(dim);
	int signum;

	// LU decomposition.
	const int status = gsl_linalg_LU_decomp(&m.matrix, p, &signum);
	if (!status)
	{
		std::cout << "L/U = " << std::endl;
		print_gsl_matrix(&m.matrix, dim, dim);
	}
	else
		std::cerr << "error: " << gsl_strerror(status) << std::endl;

#if 0
	// upper triangular matix, U.
	{
		gsl_matrix *U = gsl_matrix_alloc(dim, dim);
		gsl_matrix_set_zero(U);

		for (int i = 0; i < dim; ++i)
			for (int j = i; j < dim; ++j)
				gsl_matrix_set(U, i, j, gsl_matrix_get(&m.matrix, i, j));

		std::cout << "U = " << std::endl;
		print_gsl_matrix(U, dim, dim);

		// inverse of U.
		{
			gsl_matrix *U_inv = gsl_matrix_alloc(dim, dim);

			gsl_linalg_LU_invert(U, p, U_inv);

			std::cout << "inv(U) = " << std::endl;
			print_gsl_matrix(U_inv, dim, dim);

			gsl_matrix_free(U_inv);
		}

		gsl_matrix_free(U);
	}
#endif 

#if 0
	// lower triangular matix, L.
	{
		gsl_matrix *L = gsl_matrix_alloc(dim, dim);
		gsl_matrix_set_zero(L);

		for (int i = 0; i < dim; ++i)
		{
			for (int j = 0; j < i; ++j)
				gsl_matrix_set(L, i, j, gsl_matrix_get(&m.matrix, i, j));
			gsl_matrix_set(L, i, i, 1.0);
		}
		
		std::cout << "L = " << std::endl;
		print_gsl_matrix(L, dim, dim);

		gsl_matrix_free(L);
	}
#endif

	// solve linear system.
	{
		double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
		gsl_vector_view b = gsl_vector_view_array(b_data, dim);

		gsl_vector *x = gsl_vector_alloc(dim);

		const int status = gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);
		if (!status)
		{
			std::cout << "x = " << std::endl;
			//gsl_vector_fprintf(stdout, x, "%g");
			print_gsl_vector(x, dim);
		}
		else
			std::cerr << "error: " << gsl_strerror(status) << std::endl;

		gsl_vector_free(x);
	}

	// inverse of matrix.
	{
		gsl_matrix *a_inv = gsl_matrix_alloc(dim, dim);

		const int status = gsl_linalg_LU_invert(&m.matrix, p, a_inv);
		if (!status)
		{
			std::cout << "inverse = " << std::endl;
			print_gsl_matrix(a_inv, dim, dim);
		}
		else
			std::cerr << "error: " << gsl_strerror(status) << std::endl;

		gsl_matrix_free(a_inv);
	}

	// determinant of matrix.
	{
		const double a_det = gsl_linalg_LU_det(&m.matrix, signum);

		std::cout << "det = " << a_det << std::endl;
	}

	gsl_permutation_free(p);
}

}  // namespace my_gsl
