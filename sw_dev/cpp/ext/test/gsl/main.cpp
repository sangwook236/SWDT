//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <gsl/gsl_matrix.h>
#include <iostream>


int main(int argc, char *argv[])
{
	void vector_operation();
	void matrix_operation();
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
	void quadratic_equation_root_finding();
	void cubic_equation_root_finding();
	void polynomial_root_finding();
	void one_dim_root_finding();
	void multidim_root_finding();
	void fft();
	void random_sample();
	void distribution();
	void monte_carlo_integration();
	void simulated_annealing();

	try
	{
		//vector_operation();
		//matrix_operation();

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

		//quadratic_equation_root_finding();
		//cubic_equation_root_finding();
		//polynomial_root_finding();
		//one_dim_root_finding();
		//multidim_root_finding();
		
		//fft();
		
		random_sample();
		//distribution();
		//monte_carlo_integration();
		//simulated_annealing();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred: " << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
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
