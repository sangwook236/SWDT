//#include "stdafx.h"
#include <shogun/base/init.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

void ci_classification_example();
void gp_regression_example();

}  // namespace my_shogun

int shogun_main(int argc, char *argv[])
{
	shogun::init_shogun_with_defaults();

	std::cout << "shogun library: conjugate index classification example ----------" << std::endl;
	my_shogun::ci_classification_example();

	std::cout << "\nshogun library: Gaussian process (GP) regression example --------" << std::endl;
	my_shogun::gp_regression_example();

	shogun::exit_shogun();

	return 0;
}
