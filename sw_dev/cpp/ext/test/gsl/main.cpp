#include "stdafx.h"
#include <gsl/gsl_matrix.h>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

int wmain()
{
	void vector_operation();
	void matrix_operation();
	void polynomial_roots();
	void lu();
	void qr();
	void cholesky();
	void eigensystem();
	void svd();
	void pca();
	void levenberg_marquardt();
	void conjugate_gradient();
	void multidim_minimization_simplex();
	void multidim_minimization_steepest_descent();
	void fft();
	void random_sample();
	void distribution();
	void monte_carlo_integration();
	void simulated_annealing();

	vector_operation();
	//matrix_operation();
	//polynomial_roots();
	//lu();
	//qr();
	//cholesky();
	//eigensystem();
	//svd();
	//pca();
	//levenberg_marquardt();
	//conjugate_gradient();
	//multidim_minimization_simplex();
	//multidim_minimization_steepest_descent();
	//fft();
	//random_sample();
	//distribution();
	//monte_carlo_integration();
	//simulated_annealing();

	std::cout << "press any key to terminate" << std::flush;
	std::cin.get();

    return 0;
}

void print_gsl_vector(gsl_vector* vec, const int dim)
{
	std::cout << "[ ";
	for (int i = 0; i < dim; i++)
	{
		if (0 != i) std::cout << ' ';
		std::cout << gsl_vector_get(vec, i);
	}
	std::cout << " ]\n";
}

void print_gsl_matrix(gsl_matrix* mat, const int rdim, const int cdim)
{
	for (int i = 0; i < rdim; i++)
	{
		gsl_vector_view rvec = gsl_matrix_row(mat, i);
		print_gsl_vector(&rvec.vector, cdim);
	}
}

void print_gsl_vector(gsl_vector* vec)
{
	print_gsl_vector(vec, (int)vec->size);
}

void print_gsl_matrix(gsl_matrix* mat)
{
	print_gsl_matrix(mat, (int)mat->size1, (int)mat->size2);
}
