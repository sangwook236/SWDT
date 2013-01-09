//#include "stdafx.h"
#include <gsl/gsl_eigen.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void print_gsl_vector(gsl_vector *vec, const int dim);
void print_gsl_matrix(gsl_matrix *mat, const int rdim, const int cdim);

void eigensystem()
{
	const int dim = 4;
	double data[] = {
		1.0 , 1/2.0, 1/3.0, 1/4.0,
		1/2.0, 1/3.0, 1/4.0, 1/5.0,
		1/3.0, 1/4.0, 1/5.0, 1/6.0,
		1/4.0, 1/5.0, 1/6.0, 1/7.0
	};
	gsl_matrix_view m = gsl_matrix_view_array(data, dim, dim);
	gsl_vector *eval = gsl_vector_alloc(dim);
	gsl_matrix *evec = gsl_matrix_alloc(dim, dim);
	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(dim);

	//
	gsl_eigen_symmv(&m.matrix, eval, evec, w);
	gsl_eigen_symmv_free(w);

	// GSL_EIGEN_SORT_VAL_ASC, GSL_EIGEN_SORT_VAL_DESC, GSL_EIGEN_SORT_ABS_ASC, GSL_EIGEN_SORT_ABS_DESC
	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);

	std::cout << "eigenvalues = \n";
	print_gsl_vector(eval, dim);
	std::cout << "eigenvectors = \n";
	print_gsl_matrix(evec, dim, dim);

	gsl_vector_free(eval);
	gsl_matrix_free(evec);
}

}  // namespace my_gsl
