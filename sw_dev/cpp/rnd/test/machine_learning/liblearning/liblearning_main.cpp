//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_liblearning {

void deep_learning_example();

}  // namespace my_liblearning

int liblearning_main(int argc, char *argv[])
{
	std::cout << "liblearning library: deep learning example ----------------------" << std::endl;
	my_liblearning::deep_learning_example();  // not yet implemented.

	return 0;
}
