#include <iostream>
#include <complex>


namespace {
namespace local {
	
}  // namespace local
}  // unnamed namespace

void complex()
{
	std::complex<double> c1(1, 2), c2(-2, 3);
	std::cout << c1 << std::endl;
	std::cout << c1 + c2 << ", " << c1 - c2 << ", " << c1 * c2 << ", " << c1 / c2 << std::endl;

	std::cout << std::sin(c1) << ", " << std::cos(c1) << ", " << std::tan(c1) << std::endl;
	std::cout << std::sinh(c1) << ", " << std::cosh(c1) << ", " << std::tanh(c1) << std::endl;
	std::cout << std::sqrt(c1) << ", " << std::exp(c1) << ", " << std::log(c1) << std::endl;
}
