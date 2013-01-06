//#include "stdafx.h"
#include <gsl/gsl_linalg.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace gsl {

void print_gsl_vector(gsl_vector *vec, const int dim);
void print_gsl_matrix(gsl_matrix *mat, const int rdim, const int cdim);

void qr()
{
	const int rdim = 4, cdim = 4;
	double a_data[] = {
		0.18, 0.60, 0.57, 0.96,
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.51, 0.13, 0.19, 0.85
	};
	double b_data[] = { 1.0, 2.0, 3.0, 4.0 };
	gsl_matrix_view m = gsl_matrix_view_array(a_data, rdim, cdim);
	gsl_vector_view b = gsl_vector_view_array(b_data, rdim);
	gsl_permutation *p = gsl_permutation_alloc(cdim);
	gsl_vector *tau = gsl_vector_alloc(std::min(rdim, cdim));
	gsl_vector *norm = gsl_vector_alloc(cdim);
	gsl_vector *x = gsl_vector_alloc(cdim);
	int signum;

	gsl_matrix* Q = gsl_matrix_alloc(rdim, rdim);
	gsl_matrix* R = gsl_matrix_alloc(rdim, cdim);

	// QR decomposition (type 2)
	gsl_linalg_QRPT_decomp2(&m.matrix, Q, R, tau, p, &signum, norm);

	std::cout << "Q = \n";
	print_gsl_matrix(Q, rdim, rdim);
	std::cout << "R = \n";
	print_gsl_matrix(R, rdim, cdim);

	// solve linear system (type 2)
	gsl_linalg_QRPT_QRsolve(Q, R, p, &b.vector, x);

	std::cout << "x = \n";
	print_gsl_vector(x, cdim);

	// QR decomposition (type 1)
	gsl_linalg_QRPT_decomp(&m.matrix, tau, p, &signum, norm);

	std::cout << "Q/R = \n";
	print_gsl_matrix(&m.matrix, rdim, rdim);
	std::cout << "tau = \n";
	print_gsl_vector(tau, std::min(rdim, cdim));

	// solve linear system (type 1)
	gsl_linalg_QRPT_solve(&m.matrix, tau, p, &b.vector, x);

	std::cout << "x = \n";
	print_gsl_vector(x, cdim);

	gsl_matrix_free(Q);
	gsl_matrix_free(R);
	gsl_vector_free(x);
	gsl_vector_free(norm);
	gsl_vector_free(tau);
	gsl_permutation_free(p);
}

}  // namespace gsl
