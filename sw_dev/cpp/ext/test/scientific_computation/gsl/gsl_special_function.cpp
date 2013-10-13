//#include "stdafx.h"
#include <gsl/gsl_sf_lambert.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

void lambert_w_function()
{
	//const double x = -10.0;  // run-time error.
	//const double x = -1.0;  // run-time error.
	//const double x = -0.5;  // run-time error.
	const double x = -0.1;

	// the principal branch of the Lambert W function, W_0(x).
	//	W > -1 for x < 0.
	const double W0 = gsl_sf_lambert_W0(x);
	gsl_sf_result W0e;
	const int retval1 = gsl_sf_lambert_W0_e(x, &W0e);

	std::cout << "the principal branch of the Lambert W function, W0(x) = " << W0 << std::endl;
	std::cout << "the principal branch of the Lambert W function, W0(x) = " << W0e.val << ", error = " << W0e.err << std::endl;

	// the secondary real-valued branch of the LambertW function, W_−1(x).
	//	W < -1 for x < 0.
	const double Wm1 = gsl_sf_lambert_Wm1(x);
	gsl_sf_result Wm1e;
	const int retval2 = gsl_sf_lambert_Wm1_e(x, &Wm1e);

	std::cout << "the secondary real-valued branch of the Lambert W function, Wm1(x) = " << Wm1 << std::endl;
	std::cout << "the secondary real-valued branch of the Lambert W function, Wm1(x) = " << Wm1e.val << ", error = " << Wm1e.err << std::endl;
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
