//#include "stdafx.h"
#if defined(__HUGE)
#error error
#endif
#include <cstddef>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

/*
	Computation of the integral,
	I = int (dx dy dz)/(2pi)^3 1/(1-cos(x)cos(y)cos(z)) over (-pi,-pi,-pi) to (+pi, +pi, +pi).
	The exact answer is Gamma(1/4)^4/(4 pi^3).
	This example is taken from C.Itzykson, J.M.Drouffe, "Statistical Field Theory - Volume 1", Section 1.1, p21,
	which cites the original paper M.L.Glasser, I.J.Zucker, Proc.Natl.Acad.Sci.USA 741800 (1977)
*/
/*
	For simplicity we compute the integral over the region (0,0,0) -> (pi,pi,pi) and multiply by 8
*/

const double exact = 1.3932039296856768591842462603255;

double g(double *k, size_t dim, void *params)
{
	double A = 1.0 / (M_PI * M_PI * M_PI);
	return A / (1.0 - std::cos(k[0]) * std::cos(k[1]) * std::cos(k[2]));
}

void display_results(char *title, double result, double error)
{
	std::cout << "%s ==================" << title << std::endl;
	std::cout << "result = " << result << std::endl;
	std::cout << "sigma = " << error << std::endl;
	std::cout << "exact = " << exact << std::endl;
	std::cout << "error = " << (result - exact) << " = " << (std::fabs(result - exact) / error) << " sigma" << std::endl;
}

}  // namespace local
}  // unnamed namespace

void monte_carlo_integration()
{
	double res, err;
	double xl[3] = { 0, 0, 0 };
	double xu[3] = { M_PI, M_PI, M_PI };

	gsl_monte_function G = { &local::g, 3, NULL };
	const size_t calls = 500000;

	gsl_rng_env_setup();
	const gsl_rng_type *T = gsl_rng_default;
	gsl_rng *r = gsl_rng_alloc(T);

	//
	{
		gsl_monte_plain_state *s = gsl_monte_plain_alloc(3);
		gsl_monte_plain_integrate(&G, xl, xu, 3, calls, r, s, &res, &err);
		gsl_monte_plain_free(s);
		local::display_results("plain", res, err);
	}

	//
	{
		gsl_monte_miser_state *s = gsl_monte_miser_alloc(3);
		gsl_monte_miser_integrate(&G, xl, xu, 3, calls, r, s, &res, &err);
		gsl_monte_miser_free(s);
		local::display_results("miser", res, err);
	}

	//
	{
		gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(3);
		gsl_monte_vegas_integrate(&G, xl, xu, 3, 10000, r, s, &res, &err);
		local::display_results("vegas warm-up", res, err);

		std::cout << "converging..." << std::endl;
		do
		{
			gsl_monte_vegas_integrate(&G, xl, xu, 3, calls/5, r, s, &res, &err);
			std::cout << "result = " << res << " sigma = " << err << " chisq/dof = " << s->chisq << std::endl;
		} while (std::fabs(s->chisq - 1.0) > 0.5);

		local::display_results("vegas final", res, err);
		gsl_monte_vegas_free(s);
	}
}
