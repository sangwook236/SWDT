//#include "stdafx.h"
#include <gsl/gsl_linalg.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void lu()
{
	void print_gsl_vector(gsl_vector* vec, const int dim);
	void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim);

	const int dim = 4;
	double a_data[] = {
		0.18, 0.60, 0.57, 0.96,
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.51, 0.13, 0.19, 0.85
	};
	double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
	gsl_matrix_view m = gsl_matrix_view_array(a_data, dim, dim);
	gsl_vector_view b = gsl_vector_view_array(b_data, dim);
	gsl_permutation *p = gsl_permutation_alloc(dim);
	gsl_vector *x = gsl_vector_alloc(dim);
	gsl_matrix *a_inv = gsl_matrix_alloc(dim, dim);
	int signum;

	// LU decomposition
	gsl_linalg_LU_decomp(&m.matrix, p, &signum);

	std::cout << "L/U = \n";
	print_gsl_matrix(&m.matrix, dim, dim);

	// solve linear system
	gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);

	std::cout << "x = \n";
	//gsl_vector_fprintf(stdout, x, "%g");
	print_gsl_vector(x, dim);

	// inverse of matrix
	gsl_linalg_LU_invert(&m.matrix, p, a_inv);

	std::cout << "inverse = \n";
	print_gsl_matrix(a_inv, dim, dim);

	// determinant of matrix
	const double a_det = gsl_linalg_LU_det(&m.matrix, signum);

	std::cout << "det = " << a_det << '\n';

	gsl_matrix_free(a_inv);
	gsl_vector_free(x);
	gsl_permutation_free(p);
}
