//#include "stdafx.h"
#include <gsl/gsl_poly.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void quadratic_equation_root_finding()
{
	{
		// a * x^2 + b * x + c = 0
		const double a = 1.0, b = -3.0, c = 2.0;
		double x0, x1;
		gsl_poly_solve_quadratic(a, b, c, &x0, &x1);

		std::cout << "solution #0 = " << x0 << std::endl;
		std::cout << "solution #1 = " << x1 << std::endl;
	}

	{
		// a * x^2 + b * x + c = 0
		const double a = 1.0, b = -3.0, c = -2.0;
		gsl_complex z0, z1;
		gsl_poly_complex_solve_quadratic(a, b, c, &z0, &z1);

		std::cout << "solution #0 = " << z0.dat[0] << " + i * " << z0.dat[1] << std::endl;
		std::cout << "solution #1 = " << z1.dat[0] << " + i * " << z1.dat[1] << std::endl;
	}
}

void cubic_equation_root_finding()
{
	{
		// x^3 + b * x^2 + c * x + d = 0
		const double b = -6.0, c = 11.0, d = -6.0;
		double x0, x1, x2;
		gsl_poly_solve_cubic(b, c, d, &x0, &x1, &x2);

		std::cout << "solution #0 = " << x0 << std::endl;
		std::cout << "solution #1 = " << x1 << std::endl;
		std::cout << "solution #2 = " << x2 << std::endl;
	}

	{
		// x^3 + b * x^2 + c * x + d = 0
		const double b = -6.0, c = 11.0, d = 6.0;
		gsl_complex z0, z1, z2;
		gsl_poly_complex_solve_cubic(b, c, d, &z0, &z1, &z2);

		std::cout << "solution #0 = " << z0.dat[0] << " + i * " << z0.dat[1] << std::endl;
		std::cout << "solution #1 = " << z1.dat[0] << " + i * " << z1.dat[1] << std::endl;
		std::cout << "solution #2 = " << z2.dat[0] << " + i * " << z2.dat[1] << std::endl;
	}
}

void polynomial_root_finding()
{
	// coefficients of P(x) = -1 + x^5
	double a[6] = { -1, 0, 0, 0, 0, 1 };
	double z[10];

	gsl_poly_complex_workspace *w = gsl_poly_complex_workspace_alloc(6);

	gsl_poly_complex_solve(a, 6, w, z);

	gsl_poly_complex_workspace_free(w);

	for (int i = 0; i < 5; ++i)
		std::cout << "solution #" << i << " = " << z[2*i] << " + i * " << z[2*i+1] << std::endl;
}
