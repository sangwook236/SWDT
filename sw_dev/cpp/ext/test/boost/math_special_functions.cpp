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


namespace {
namespace local {

void gamma_functions()
{
}

void factorial()
{
}

void binomial_coefficients()
{
}

void beta_functions()
{
}

void error_functions()
{
}

void legendre_polynomials()
{
}

void laguerre_polynomials()
{
}

void hermite_polynomials()
{
}

void spheric_harmonic()
{
}

void bessel_functions()
{
}

void elliptic_integrals()
{
}

void zeta_functions()
{
}

void exponential_integrals()
{
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
