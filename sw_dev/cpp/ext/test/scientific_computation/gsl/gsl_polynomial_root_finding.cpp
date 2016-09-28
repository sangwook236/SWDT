//#include "stdafx.h"
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void quadratic_equation_root_finding()
{
	{
		// a * x^2 + b * x + c = 0.
		const double a = 1.0, b = -3.0, c = 2.0;
		double x0, x1;
		const int numRoots = gsl_poly_solve_quadratic(a, b, c, &x0, &x1);

		std::cout << "#solutions = " << numRoots << std::endl;
		if (0 == numRoots)
			std::cout << "\tReal root not found" << std::endl;
		else if (1 == numRoots)
			std::cout << "\tSolution #0 = " << x0 << std::endl;
		else if (2 == numRoots)
		{
			std::cout << "\tSolution #0 = " << x0 << std::endl; 
			std::cout << "\tSolution #1 = " << x1 << std::endl;
		}
		else std::cerr << "\tRoot finding failed" << std::endl;
	}

	{
		// a * x^2 + b * x + c = 0.
		const double a = 1.0, b = -3.0, c = -2.0;
		gsl_complex z0, z1;
		const int numRoots = gsl_poly_complex_solve_quadratic(a, b, c, &z0, &z1);

		std::cout << "#complex solutions = " << numRoots << std::endl;
		if (1 == numRoots)  // Real root.
			std::cout << "\tSolution #0 = " << z0.dat[0] << " + i * " << z0.dat[1] << std::endl;
		else if (2 == numRoots)
		{
			std::cout << "\tSolution #0 = " << z0.dat[0] << " + i * " << z0.dat[1] << std::endl;
			std::cout << "\tSolution #1 = " << z1.dat[0] << " + i * " << z1.dat[1] << std::endl;
		}
		else std::cerr << "\tRoot finding failed" << std::endl;
	}
}

void cubic_equation_root_finding()
{
	{
		// x^3 + b * x^2 + c * x + d = 0.
		const double b = -6.0, c = 11.0, d = -6.0;
		double x0, x1, x2;
		const int numRoots = gsl_poly_solve_cubic(b, c, d, &x0, &x1, &x2);

		std::cout << "#solutions = " << numRoots << std::endl;
		if (1 == numRoots) std::cout << "\tSolution #0 = " << x0 << std::endl;
		else if (3 == numRoots)
		{
			std::cout << "\tSolution #0 = " << x0 << std::endl;
			std::cout << "\tSolution #1 = " << x1 << std::endl;
			std::cout << "\tSolution #2 = " << x2 << std::endl;
		}
		else std::cerr << "\tRoot finding failed" << std::endl;
	}

	{
		// x^3 + b * x^2 + c * x + d = 0.
		const double b = -6.0, c = 11.0, d = 6.0;
		gsl_complex z0, z1, z2;
		const int numRoots = gsl_poly_complex_solve_cubic(b, c, d, &z0, &z1, &z2);

		std::cout << "#complex solutions = " << numRoots << std::endl;
		if (3 == numRoots)
		{
			std::cout << "\tSolution #0 = " << z0.dat[0] << " + i * " << z0.dat[1] << std::endl;
			std::cout << "\tSolution #1 = " << z1.dat[0] << " + i * " << z1.dat[1] << std::endl;
			std::cout << "\tSolution #2 = " << z2.dat[0] << " + i * " << z2.dat[1] << std::endl;
		}
		else std::cerr << "\tRoot finding failed" << std::endl;
	}
}

void polynomial_root_finding()
{
	// Coefficients of P(x) = -1 + x^5.
	const size_t degree = 5;
	double a[degree + 1] = { -1, 0, 0, 0, 0, 1 };
	double z[degree * 2];

	gsl_poly_complex_workspace *w = gsl_poly_complex_workspace_alloc(6);

	const int retval = gsl_poly_complex_solve(a, 6, w, z);

	gsl_poly_complex_workspace_free(w);

	if (GSL_SUCCESS == retval)
	{
		std::cout << "Roots found:" << std::endl;
		for (int i = 0; i < degree; ++i)
			std::cout << "\tSolution #" << i << " = " << z[2 * i] << " + i * " << z[2 * i + 1] << std::endl;
	}
	else
		std::cout << "Root finding failed" << std::endl;
}

}  // namespace my_gsl
