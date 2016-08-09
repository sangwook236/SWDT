#include <particleplusplus/setting.h>
#include <particleplusplus/pfilter.h>
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <boost/chrono.hpp>
#else
#include <chrono>
#endif
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <cmath>


namespace {
namespace local {

const precision_type PI = boost::math::constants::pi<double>();
const precision_type alpha = 0.91;
const precision_type beta = 1.0;

typedef long double state_type;
typedef long double obsv_type;

// initialize the random seed.
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
const unsigned long seed = boost::chrono::system_clock::now().time_since_epoch().count();
#else
const unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
#endif
std::default_random_engine generator(seed);

std::normal_distribution<state_type> distribution(0.0, 1.0);

// transition/motion/action model.
// long double f(state_type x_n, state_type x_{n-1})
precision_type f(state_type x1, state_type x2)
{
	// f(x_n | x_{n-1}) = normal_distribution(x_n ; alpha * x_{n-1}, 1)
	return std::exp(-0.5 * std::pow(x1 - alpha * x2, 2));
}

// measurement/perception/sensor/observation model.
// long double g(state_type x_n, obsv_type y_n)
precision_type g(state_type x, obsv_type y)
{
	// g(y_n | x_n) = normal_distribution(y_n ; 0, beta^2 * exp(x_n))
	return 1.0 / std::exp(x / 2.0) * std::exp(-0.5 * std::pow(y / beta / std::exp(x / 2.0), 2));
}

// proposal distribution.
// long double q(state_type x_n, state_type x_{n-1}, obsv_type y)
precision_type q(state_type x1, state_type x2, obsv_type y)
{
	return std::exp(-0.5 * std::pow(x1 - alpha * x2, 2));
}

// proposal sampling function.
// state_type q_sam(state_type x_{n-1}, obsv_type y)
state_type q_sam(state_type x, obsv_type y)
{
	return distribution(generator) + alpha * x;
}

// [ref] ${PARTICLEPLUSPLUS_HOME}/main.cpp
void basic_sample()
{
	const std::string input_filename("./data/bayesian_filtering/data_y.dat");
	const std::string output_filename("./data/bayesian_filtering/data_xhat.dat");

	std::ifstream in_stream(input_filename);
	if (!in_stream.is_open())
	{
		std::cerr << "input file not found: " << output_filename << std::endl;
		return;
	}

	pfilter<state_type, obsv_type> particleFilter(f, g, q, q_sam);
	in_stream >> particleFilter;

	particleFilter.initialize(2000);  // initialize with the number of particles we want to use.

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const boost::chrono::high_resolution_clock::time_point t1 = boost::chrono::high_resolution_clock::now();
#else
	const std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
	particleFilter.iterate();  // run
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const boost::chrono::high_resolution_clock::time_point t2 = boost::chrono::high_resolution_clock::now();
#else
	const std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
#endif

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	const boost::chrono::duration<double> time_span = boost::chrono::duration_cast<boost::chrono::duration<double> >(t2 - t1);
#else
	const std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
#endif
	std::cout << "it took " << time_span.count() << " seconds." << std::endl;

	//
	std::ofstream out_stream(output_filename);
	if (!out_stream.is_open())
	{
		std::cerr << "output file not found: " << output_filename << std::endl;
		return;
	}

	out_stream << particleFilter;  // output data.
}

}  // namespace local
}  // unnamed namespace

namespace my_particleplusplus {

}  // namespace my_particleplusplus

int particleplusplus_main(int argc, char *argv[])
{
	local::basic_sample();

	return 0;
}
