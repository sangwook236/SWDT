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

void cholesky()
{
	const int dim = 4;
	double a_data[] = {
		0.4802, 0.3147, 0.7412, 0.9365,
		0.3147, 0.5245, 0.8953, 1.0237,
		0.7412, 0.8953, 2.2820, 1.9231,
		0.9365, 1.0237, 1.9231, 2.4161,
	};
	gsl_matrix_view m = gsl_matrix_view_array(a_data, dim, dim);

	// Cholesky decomposition.
	{
		gsl_linalg_cholesky_decomp(&m.matrix);
	
		std::cout << "L/L^T = " << std::endl;
		print_gsl_matrix(&m.matrix, dim, dim);

		// caution: m is changed here.
	}

	// solve linear system.
	{
		double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
		gsl_vector_view b = gsl_vector_view_array(b_data, dim);
		gsl_vector *x = gsl_vector_alloc(dim);

		gsl_linalg_cholesky_solve(&m.matrix, &b.vector, x);

		std::cout << "x = " << std::endl;
		print_gsl_vector(x, dim);

		gsl_vector_free(x);
	}

	// inverse of matrix.
	{
		gsl_linalg_cholesky_invert(&m.matrix);

		std::cout << "inverse = " << std::endl;
		print_gsl_matrix(&m.matrix, dim, dim);
	}
}

}  // namespace my_gsl
