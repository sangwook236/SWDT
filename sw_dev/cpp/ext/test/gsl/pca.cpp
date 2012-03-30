//#include "stdafx.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>
#include <iostream>
#include <cassert>


namespace {
namespace local {

void pca_by_svd(gsl_matrix* U, const int rdim, const int cdim)
{
	void print_gsl_vector(gsl_vector* vec, const int dim);
	void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim);

	assert(rdim >= cdim);

	const int min_dim = std::min(rdim, cdim);
	//const int max_dim = std::max(rdim, cdim);

	gsl_matrix* V = gsl_matrix_alloc(cdim, cdim);
	gsl_vector* S = gsl_vector_alloc(min_dim);
	gsl_vector* work = gsl_vector_alloc(min_dim);

	//
	gsl_linalg_SV_decomp(U, V, S, work);
	gsl_vector_free(work);

	std::cout << "U = \n";
	print_gsl_matrix(U, rdim, cdim);
	//std::cout << "V = \n";
	//print_gsl_matrix(V, cdim, cdim);
	std::cout << "S = \n";
	print_gsl_vector(S, min_dim);

	gsl_matrix_free(V);
	gsl_vector_free(S);
}

void pca_by_eigen(gsl_matrix* mat, const int rdim, const int cdim)
{
	void print_gsl_vector(gsl_vector* vec, const int dim);
	void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim);

	const int dim = rdim;

	gsl_matrix* m = gsl_matrix_alloc(dim, dim);
	gsl_matrix_set_zero(m);
	gsl_blas_dgemm(
		CblasNoTrans, CblasTrans,
		1.0, mat, mat,
		0.0, m
	);
	//print_gsl_matrix(m, dim, dim);

	//
	gsl_vector* eval = gsl_vector_alloc(dim);
	gsl_matrix* evec = gsl_matrix_alloc(dim, dim);
	gsl_eigen_symmv_workspace* w = gsl_eigen_symmv_alloc(dim);

	gsl_eigen_symmv(m, eval, evec, w);
	gsl_matrix_free(m);

	gsl_eigen_symmv_free(w);
	//gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_DESC);

	std::cout << "eigenvalues = \n";
	print_gsl_vector(eval, dim);
	std::cout << "eigenvectors = \n";
	print_gsl_matrix(evec, dim, dim);

	gsl_vector_free(eval);
	gsl_matrix_free(evec);
}

}  // namespace local
}  // unnamed namespace

void pca()
{
	void print_gsl_vector(gsl_vector* vec, const int dim);
	void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim);

	const int rdim = 9, cdim = 4;
	double data[] = {
		53.50,		-161.50,	24.50,		83.50,
		52.50,		42.50,		-141.50,	46.50,
		-84.50,		-108.50,	101.50,		91.50,
		2.75,		6.75,		-16.25,		6.75,
		5.50,		-9.50,		31.50,		-27.50,
		-8.00,		1.00,		-2.00,		9.00,
		-127.25,	-110.25,	115.75,		121.75,
		49.50,		49.50,		-148.50,	49.50,
		47.00,		-168.00,	56.00,		65.00,
	};

	gsl_matrix_view mat = gsl_matrix_view_array(data, rdim, cdim);
	//print_gsl_matrix(&mat.matrix, rdim, cdim);

	gsl_matrix* mat2 = gsl_matrix_alloc(rdim, cdim);
	gsl_matrix_memcpy(mat2, &mat.matrix);

	local::pca_by_svd(&mat.matrix, rdim, cdim);
	local::pca_by_eigen(mat2, rdim, cdim);

	gsl_matrix_free(mat2);
}
