#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mcmcpp {

void normal_example();
void binomial_example();
void multinomial_example();
void normal_mixture_example();

void multi_scan_mcmcda_algorithm();

}  // namespace my_mcmcpp

int mcmcpp_main(int argc, char *argv[])
{
	// Example -------------------------------------------------------------
	{
		my_mcmcpp::normal_example();
		//my_mcmcpp::binomial_example();
		//my_mcmcpp::multinomial_example();

		//my_mcmcpp::normal_mixture_example();  // TODO [check] >> Not working ???
	}

	// Application ---------------------------------------------------------
	{
		//my_mcmcpp::multi_scan_mcmcda_algorithm();  // Not yet implemented.
	}

	return 0;
}
