#include "stdafx.h"
#include <gsl/gsl_linalg.h>
#include <iostream>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


void svd()
{
	void print_gsl_vector(gsl_vector* vec, const int dim);
	void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim);
/*
	const int rdim = 4, cdim = 4;
	double a_data[] = {
		0.18, 0.60, 0.57, 0.96,
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.51, 0.13, 0.19, 0.85
	};
	double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
*/
	const int rdim = 5, cdim = 4;
	double a_data[] = {
		0.18, 0.60, 0.57, 0.96, 
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.51, 0.13, 0.19, 0.85,
		0.41, 0.13, 0.57, 0.24,
	};
	double b_data[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };

	const int min_dim = std::min(rdim, cdim);
	//const int max_dim = std::max(rdim, cdim);

	gsl_matrix_view A = gsl_matrix_view_array(a_data, rdim, cdim);
	gsl_vector_view b = gsl_vector_view_array(b_data, rdim);
	gsl_vector* x = gsl_vector_alloc(cdim);

	//gsl_matrix* U = gsl_matrix_alloc(rdim, rdim);
	gsl_matrix* U = gsl_matrix_alloc(rdim, cdim);
	gsl_matrix* V = gsl_matrix_alloc(cdim, cdim);
	gsl_vector* S = gsl_vector_alloc(min_dim);
	gsl_vector* work = gsl_vector_alloc(min_dim);

	gsl_matrix_memcpy(U, &A.matrix);

	// SVD
	gsl_linalg_SV_decomp(U, V, S, work);
	//gsl_linalg_SV_decomp_mod(U, Xwork, V, S, work);
	//gsl_linalg_SV_decomp_jacobi(U, V, S);
	gsl_vector_free(work);

	std::cout << "U = \n";
	//gsl_matrix_fprintf(stdout, U, "%g");
	//print_gsl_matrix(U, rdim, rdim);
	print_gsl_matrix(U, rdim, cdim);
	std::cout << "V = \n";
	//gsl_matrix_fprintf(stdout, V, "%g");
	print_gsl_matrix(V, cdim, cdim);
	std::cout << "S = \n";
	//gsl_vector_fprintf(stdout, S, "%g");
	print_gsl_vector(S, min_dim);

	// solve linear system
	gsl_linalg_SV_solve(U, V, S, &b.vector, x);

	std::cout << "x = \n";
	//gsl_vector_fprintf(stdout, x, "%g");
	print_gsl_vector(x, cdim);

	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_vector_free(S);
	gsl_vector_free(x);
}
