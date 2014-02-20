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

}  // namespace my_mcmcpp

int mcmcpp_main(int argc, char *argv[])
{
	// example -------------------------------------------------------------
	{
		//my_mcmcpp::normal_example();
		//my_mcmcpp::binomial_example();
		//my_mcmcpp::multinomial_example();

		my_mcmcpp::normal_mixture_example();  // TODO [check] >> not working ???
	}

	// application ---------------------------------------------------------
	{
	}

	return 0;
}
