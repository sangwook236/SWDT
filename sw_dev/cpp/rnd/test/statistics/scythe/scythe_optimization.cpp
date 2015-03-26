//#include "stdafx.h"
#if !defined(__FUNCTION__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __FUNCTION__ L""
//#else
#define __FUNCTION__ ""
//#endif
#endif
#if !defined(__func__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __func__ L""
//#else
#define __func__ ""
//#endif
#endif

#include <scythestat/optimize.h>
#include <iostream>


namespace {
namespace local {

double x_cubed_plus_2x_a(double x)
{
	return (x * x * x + 2 * x);
}

struct x_cubed_plus_2x_b
{
	double operator()(double x) const
	{
		return (x * x * x + 2 * x);
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_scythe {

// [ref] "The Scythe Statistical Library: An Open Source C++ Library for Statistical Computation", Daniel Pemstein, Kevin M. Quinn, and Andrew D. Martin, JSS 2011.
void optimization()
{
	// Numerical utilities.
	{
		std::cout << scythe::adaptsimp(local::x_cubed_plus_2x_a, 0.0, 4.0, 10) << std::endl;
		std::cout << scythe::adaptsimp(local::x_cubed_plus_2x_b(), 0.0, 4.0, 10) << std::endl;
	}
}

}  // namespace my_scythe
