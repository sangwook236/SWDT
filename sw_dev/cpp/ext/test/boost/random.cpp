#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random.hpp>
#include <iostream>
#include <fstream>
#include <ctime>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void random_boost()
{
	typedef boost::minstd_rand base_generator_type;
	//typedef boost::mt19937 base_generator_type;
	base_generator_type baseGenerator(static_cast<unsigned int>(std::time(NULL)));

	// uniform (integer)
	{
		typedef boost::uniform_int<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		generator_type die_gen(baseGenerator, distribution_type(1, 6));
		for(int j = 0; j < 10; ++j)
		{
			//baseGenerator.seed(42u);  // NOTICE [caution] >> Generate repetitive sample.

			for(int i = 0; i < 10; ++i)
				std::cout << die_gen() << ' ';
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// uniform (real)
	{
		typedef boost::uniform_real<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		baseGenerator.seed(static_cast<unsigned int>(std::time(NULL)));

		generator_type uni_gen(baseGenerator, distribution_type(0, 1));
		for(int j = 0; j < 10; ++j)
		{
			//baseGenerator.seed(42u);  // Notice [caution] >> Generate repetitive sample.

			for (int i = 0; i < 10; ++i)
				std::cout << uni_gen() << ' ';
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// normal
	{
		typedef boost::normal_distribution<> distribution_type;
		typedef boost::variate_generator<base_generator_type &, distribution_type> generator_type;

		const double mean = 1000.0;
		const double sigma = 100.0;
		generator_type normal_gen(baseGenerator, distribution_type(mean, sigma));
		for(int j = 0; j < 10; ++j)
		{
			//baseGenerator.seed(42u);  // NOTICE [caution] >> Generate repetitive sample.

			for(int i = 0; i < 10; ++i)
				std::cout << normal_gen() << ' ';
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	//
	//base_generator_type saved_generator = baseGenerator;
	//assert(baseGenerator == saved_generator);

	//
	//std::ofstream stream("rng.saved", std::ofstream::trunc);
	//stream << baseGenerator;
}
