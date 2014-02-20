#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_bayespp {

void simple_example();
void simple_quadratic_observer_example();

void position_and_velocity_filter_example();
void position_and_velocity_SIR_filter_example();

void SLAM_example();

}  // namespace my_bayespp

int bayespp_main(int argc, char *argv[])
{
	my_bayespp::simple_example();
	//my_bayespp::simple_quadratic_observer_example();

	//my_bayespp::position_and_velocity_filter_example();
	//my_bayespp::position_and_velocity_SIR_filter_example();

	// application ---------------------------------------------------------
	//my_bayespp::SLAM_example();

	return 0;
}
