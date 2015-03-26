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

#include <scythestat/rng.h>
#include <scythestat/rng/mersenne.h>
#include <scythestat/rng/wrapped_generator.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_scythe {

// [ref] "The Scythe Statistical Library: An Open Source C++ Library for Statistical Computation", Daniel Pemstein, Kevin M. Quinn, and Andrew D. Martin, JSS 2011.
void random()
{
	{
		scythe::mersenne myrng;
		const double sum = myrng() + myrng.runif() + myrng.rnorm(0, 1) + myrng.rf(2, 50);

		std::cout << "sum = " << sum << std::endl;
	}

	{
		typedef boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > boost_twister;
		boost::mt19937 generator(42u);
		boost_twister uni(generator, boost::uniform_real<>(0, 1));
		scythe::wrapped_generator<boost_twister> wgen(uni);
		std::cout << wgen.rnorm(6, 6, 0, 1) << std::endl;
	}
}

}  // namespace my_scythe
