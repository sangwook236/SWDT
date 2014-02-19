#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_scythe {

void matrix_operation();
void random();
void optimization();
void parametric_bootstrap_example();

}  // namespace my_scythe

namespace my_scythemcmc {

void normal_example();

}  // namespace my_scythemcmc

int scythe_main(int argc, char *argv[])
{
	// Scythe Statistical Library ------------------------------------------
	{
		//my_scythe::matrix_operation();
		//my_scythe::random();
		//my_scythe::optimization();

		//my_scythe::parametric_bootstrap_example();
	}

	// Scythe MCMC library -------------------------------------------------
	{
		my_scythemcmc::normal_example();
	}

	return 0;
}
