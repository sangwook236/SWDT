#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/special_functions/ellint_3.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void gamma_functions()
{
	throw std::runtime_error("not yet implemented");
}

void factorial()
{
	throw std::runtime_error("not yet implemented");
}

void binomial_coefficients()
{
	throw std::runtime_error("not yet implemented");
}

void beta_functions()
{
	throw std::runtime_error("not yet implemented");
}

void error_functions()
{
	throw std::runtime_error("not yet implemented");
}

void legendre_polynomials()
{
	throw std::runtime_error("not yet implemented");
}

void laguerre_polynomials()
{
	throw std::runtime_error("not yet implemented");
}

void hermite_polynomials()
{
	throw std::runtime_error("not yet implemented");
}

void spheric_harmonic()
{
	throw std::runtime_error("not yet implemented");
}

void bessel_functions()
{
	// the Bessel functions of the first and second kinds
	{
		double v = 1.0, x = 1.0;

		std::cout << "the Bessel function, J_v(x) of the first kind and order " << v << " = " << boost::math::cyl_bessel_j(v, x) << std::endl;
		std::cout << "the Bessel function, Y_v(x) = N_v(x) of the second kind and order " << v << " = " << boost::math::cyl_neumann(v, x) << std::endl;
	}

	// the modified Bessel functions of the first and second kinds
	{
		double v = 1.0, x = 1.0;

		std::cout << "the modified Bessel function, I_v(x) of the first kind and order " << v << " = " << boost::math::cyl_bessel_i(v, x) << std::endl;
		std::cout << "the modified Bessel function, K_v(x) of the second kind and order " << v << " = " << boost::math::cyl_bessel_k(v, x) << std::endl;
	}

	// the spherical Bessel functions of the first and second kinds
	{
		double v = 1.0, x = 1.0;

		std::cout << "the spherical Bessel function, j_v(x) of the first kind and order " << v << " = " << boost::math::sph_bessel(v, x) << std::endl;
		std::cout << "the spherical Bessel function, y_v(x) = n_v(x) of the second kind and order " << v << " = " << boost::math::sph_neumann(v, x) << std::endl;
	}
}

void elliptic_integrals()
{
	throw std::runtime_error("not yet implemented");
}

void zeta_functions()
{
	throw std::runtime_error("not yet implemented");
}

void exponential_integrals()
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void math_special_functions()
{
	local::gamma_functions();
	local::factorial();
	local::binomial_coefficients();
	
	local::beta_functions();
	local::error_functions();
	
	local::legendre_polynomials();
	local::laguerre_polynomials();
	local::hermite_polynomials();
	local::spheric_harmonic();
	
	local::bessel_functions();
	local::elliptic_integrals();
	local::zeta_functions();
	local::exponential_integrals();
}
