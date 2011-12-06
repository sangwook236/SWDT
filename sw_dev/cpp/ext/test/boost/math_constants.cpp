#include "stdafx.h"
#include <boost/math/constants/constants.hpp>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

void math_constants()
{
	// pi
	std::cout << "pi" << std::endl;
	std::cout << '\t' << boost::math::constants::pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi<long double>() << std::endl;

	// root pi
	std::cout << "root pi" << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<long double>() << std::endl;

	// root half pi
	std::cout << "root half pi" << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<long double>() << std::endl;

	// root two pi
	std::cout << "root two pi" << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<long double>() << std::endl;

	// root ln pi
	std::cout << "root two pi" << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<long double>() << std::endl;

	// e
	std::cout << "e" << std::endl;
	std::cout << '\t' << boost::math::constants::e<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::e<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::e<long double>() << std::endl;

	// euler
	std::cout << "euler" << std::endl;
	std::cout << '\t' << boost::math::constants::euler<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::euler<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::euler<long double>() << std::endl;

	// 1 / 2
	std::cout << "1 / 2" << std::endl;
	std::cout << '\t' << boost::math::constants::half<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::half<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::half<long double>() << std::endl;

	// 1 / 3
	std::cout << "1 / 3" << std::endl;
	std::cout << '\t' << boost::math::constants::third<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::third<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::third<long double>() << std::endl;

	// 2 / 3
	std::cout << "2 / 3" << std::endl;
	std::cout << '\t' << boost::math::constants::twothirds<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::twothirds<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::twothirds<long double>() << std::endl;

	// root two
	std::cout << "root two" << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<long double>() << std::endl;

	// ln two
	std::cout << "ln two" << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<long double>() << std::endl;

	// ln ln two
	std::cout << "ln ln two" << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<long double>() << std::endl;

	// pi - 3
	std::cout << "pi - 3" << std::endl;
	std::cout << '\t' << boost::math::constants::pi_minus_three<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_minus_three<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_minus_three<long double>() << std::endl;

	// 4 - pi
	std::cout << "4 - pi" << std::endl;
	std::cout << '\t' << boost::math::constants::four_minus_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::four_minus_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::four_minus_pi<long double>() << std::endl;

	// (4 - pi)^(3 / 2)
	std::cout << "(4 - pi)^(3 / 2)" << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<long double>() << std::endl;

	// exp(-1 / 2)
	std::cout << "exp(-1 / 2)" << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<long double>() << std::endl;
}
