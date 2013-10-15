//#include "stdafx.h"
#include <gsl/gsl_sf_lambert.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void lambert_w_function()
{
	// [ref] http://en.wikipedia.org/wiki/Lambert_w_function.
	// Since W is not injective, the relation W is multivalued (except at 0).
	// If we restrict attention to real-valued W then the relation is defined only for x >= −1/e, and is double-valued on (−1/e, 0);
	// the additional constraint W >= −1 defines a single-valued function W_0(x).
	// We have W_0(0) = 0 and W_0(−1/e) = −1.
	// Meanwhile, the lower branch has W <= −1 and is denoted W_{−1}(x).
	// It decreases from W_{−1}(−1/e) = −1 to W_{−1}(0−) = −inf.

	// x >= -1/e = 0.3679.
	//const double x = -1.0 / std::exp(1.0);
	//const double x = -0.3;
	//const double x = 0.0;
	const double x = 3.0;
	//const double x = 10.0;

	// the principal branch of the Lambert W function, W_0(x).
	//	W >= -1 for x >= -1/e.
	const double W0 = gsl_sf_lambert_W0(x);
	gsl_sf_result W0e;
	const int retval1 = gsl_sf_lambert_W0_e(x, &W0e);

	std::cout << "the principal branch of the Lambert W function, W0(" << x << ") = " << W0 << std::endl;
	std::cout << "the principal branch of the Lambert W function, W0(" << x << ") = " << W0e.val << ", error = " << W0e.err << std::endl;

	// the secondary real-valued branch of the LambertW function, W_{−1}(x).
	//	W <= -1 for -1/e <= x < 0.
	//	W = W_0 for x >= 0.
	const double Wm1 = gsl_sf_lambert_Wm1(x);
	gsl_sf_result Wm1e;
	const int retval2 = gsl_sf_lambert_Wm1_e(x, &Wm1e);

	std::cout << "the secondary real-valued branch of the Lambert W function, Wm1(" << x << ") = " << Wm1 << std::endl;
	std::cout << "the secondary real-valued branch of the Lambert W function, Wm1(" << x << ") = " << Wm1e.val << ", error = " << Wm1e.err << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_gsl {

void special_function()
{
	// Lambert W function.
	local::lambert_w_function();
}

}  // namespace my_gsl
