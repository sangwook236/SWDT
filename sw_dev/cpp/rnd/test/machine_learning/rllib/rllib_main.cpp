#include <rl.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_rllib {

void cliff_walking_qlearning_example();
void cliff_walking_sarsa_example();
void boyan_chain_lstd_example();
void inverted_pendulum_lspi_example();
void inverted_pendulum_ktdq_example();
void inverted_pendulum_mlp_ktdq_example();
void mountain_car_ktdsqrsa_example();

}  // namespace my_rllib

int rllib_main(int argc, char *argv[])
{
	try
	{
		//my_rllib::cliff_walking_qlearning_example();
		//my_rllib::cliff_walking_sarsa_example();

		//my_rllib::boyan_chain_lstd_example();
		//my_rllib::inverted_pendulum_lspi_example();

		my_rllib::inverted_pendulum_ktdq_example();
		my_rllib::inverted_pendulum_mlp_ktdq_example();
		my_rllib::mountain_car_ktdsqrsa_example();
	}
	catch (const rl::exception::Any& e)
	{
		std::cerr << "rl::exception::Any caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
