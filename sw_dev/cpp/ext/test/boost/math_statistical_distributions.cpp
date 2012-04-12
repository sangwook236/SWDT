#include <boost/math/distributions/negative_binomial.hpp>  // for negative binomial distribution
using boost::math::negative_binomial_distribution;  // default type is double
using boost::math::negative_binomial;  // typedef provides default type is double
#include <boost/math/distributions/binomial.hpp>  // for binomial distribution
#include <boost/math/distributions/beta.hpp>  // for beta distribution
#include <boost/math/distributions/gamma.hpp>  // for gamma distribution
#include <boost/math/distributions/normal.hpp>  // for normal distribution
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void math_statistical_distributions()
{
	std::cout << "Example: Normal distribution, Miscellaneous Applications.";

	{
		// traditional tables and values.
		double step = 1.;  // in z
		double range = 4;  // min and max z = -range to +range.
		int precision = 17;  // traditional tables are only computed to much lower precision.

		// construct a standard normal distribution s
		boost::math::normal s;  // (default mean = zero, and standard deviation = unity)
		//boost::math::normal s(mean, std);
		std::cout << "standard normal distribution: mean = " << s.mean() << ", standard deviation = " << s.standard_deviation() << std::endl;
		std::cout << "probability distribution function values" << std::endl;
		std::cout << "  z " "      pdf " << std::endl;
		std::cout.precision(5);
		for (double z = -range; z < range + step; z += step)
		{
			std::cout << std::left << std::setprecision(3) << std::setw(6) << z << " " <<
				std::setprecision(precision) << std::setw(12) << boost::math::pdf(s, z) << std::endl;
		}
		std::cout.precision(6);  // default

		// for a standard normal distribution
		std::cout << "standard normal mean = "<< s.mean() << ", standard deviation = " << s.standard_deviation() << std::endl;
		std::cout << "integral (area under the curve) from -infinity up to z " << std::endl;
		std::cout << "  z " "      cdf " << std::endl;
		for (double z = -range; z < range + step; z += step)
		{
			std::cout << std::left << std::setprecision(3) << std::setw(6) << z << " " <<
				std::setprecision(precision) << std::setw(12) << boost::math::cdf(s, z) << std::endl;
		}
		std::cout.precision(6);  // default
	}
}
