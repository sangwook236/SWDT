#include <boost/math/constants/constants.hpp>
#include <iostream>


void math_constants()
{
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
	std::cout << '\t' << boost::math::constants::two_thirds<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_thirds<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_thirds<long double>() << std::endl;

	// 3 / 4
	std::cout << "3 / 4" << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters<long double>() << std::endl;

	// root two
	std::cout << "root two" << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two<long double>() << std::endl;

	// root three
	std::cout << "root three" << std::endl;
	std::cout << '\t' << boost::math::constants::root_three<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_three<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_three<long double>() << std::endl;

	// (root two) / 2
	std::cout << "half root two" << std::endl;
	std::cout << '\t' << boost::math::constants::half_root_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::half_root_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::half_root_two<long double>() << std::endl;

	// ln two
	std::cout << "ln two" << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_two<long double>() << std::endl;

	// ln ten
	std::cout << "ln ten" << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ten<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ten<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ten<long double>() << std::endl;

	// ln ln two
	std::cout << "ln ln two" << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::ln_ln_two<long double>() << std::endl;

	// root (ln four)
	std::cout << "root ln four" << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_ln_four<long double>() << std::endl;

	// 1 / (root 2)
	std::cout << "1 / (root two)" << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two<long double>() << std::endl;

	// pi
	std::cout << "pi" << std::endl;
	std::cout << '\t' << boost::math::constants::pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi<long double>() << std::endl;

	// pi / 2
	std::cout << "pi / 2" << std::endl;
	std::cout << '\t' << boost::math::constants::half_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::half_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::half_pi<long double>() << std::endl;

	// pi / 3
	std::cout << "pi / 3" << std::endl;
	std::cout << '\t' << boost::math::constants::third_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::third_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::third_pi<long double>() << std::endl;

	// pi / 6
	std::cout << "pi / 6" << std::endl;
	std::cout << '\t' << boost::math::constants::sixth_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::sixth_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::sixth_pi<long double>() << std::endl;

	// 2 * pi
	std::cout << "2 * pi" << std::endl;
	std::cout << '\t' << boost::math::constants::two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_pi<long double>() << std::endl;

	// 2 / 3 * pi
	std::cout << "2 / 3 * pi" << std::endl;
	std::cout << '\t' << boost::math::constants::two_thirds_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_thirds_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::two_thirds_pi<long double>() << std::endl;

	// 3 / 4 * pi
	std::cout << "3 / 4 * pi" << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::three_quarters_pi<long double>() << std::endl;

	// 4 / 3 * pi
	std::cout << "4 / 3 * pi" << std::endl;
	std::cout << '\t' << boost::math::constants::four_thirds_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::four_thirds_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::four_thirds_pi<long double>() << std::endl;

	// 1 / (2 * pi)
	std::cout << "1 / (2 * pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_two_pi<long double>() << std::endl;

	// root pi
	std::cout << "root pi" << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_pi<long double>() << std::endl;

	// root (pi / 2)
	std::cout << "root (pi / 2)" << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_half_pi<long double>() << std::endl;

	// root (2 * pi)
	std::cout << "root (2 * pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<long double>() << std::endl;

	// 1 / (root pi)
	std::cout << "1 / (root pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_pi<long double>() << std::endl;

	// 1 / (root (2 * pi))
	std::cout << "1 / (root (2 * pi))" << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_root_two_pi<long double>() << std::endl;

	// root (1 /  pi)
	std::cout << "root (1 / pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::root_one_div_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_one_div_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_one_div_pi<long double>() << std::endl;

	// root (2 * pi)
	std::cout << "root (2 * pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::root_two_pi<long double>() << std::endl;

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

	// pi^e
	std::cout << "pi^e" << std::endl;
	std::cout << '\t' << boost::math::constants::pi_pow_e<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_pow_e<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_pow_e<long double>() << std::endl;

	// pi^2
	std::cout << "pi^2" << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr<long double>() << std::endl;

	// pi^2 / 6
	std::cout << "pi^2 / 6" << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr_div_six<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr_div_six<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_sqr_div_six<long double>() << std::endl;

	// pi^3
	std::cout << "pi^3" << std::endl;
	std::cout << '\t' << boost::math::constants::pi_cubed<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_cubed<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pi_cubed<long double>() << std::endl;

	// cube root pi
	std::cout << "cube root pi" << std::endl;
	std::cout << '\t' << boost::math::constants::cbrt_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::cbrt_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::cbrt_pi<long double>() << std::endl;

	// 1 / (cube root pi)
	std::cout << "1 / (cube root pi)" << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_cbrt_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_cbrt_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::one_div_cbrt_pi<long double>() << std::endl;

	// (4 - pi)^(3 / 2)
#if 0
	std::cout << "(4 - pi)^(3 / 2)" << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::pow23_four_minus_pi<long double>() << std::endl;
#endif

	// exp(-1 / 2)
	std::cout << "exp(-1 / 2)" << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<float>() << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<double>() << std::endl;
	std::cout << '\t' << boost::math::constants::exp_minus_half<long double>() << std::endl;
}
