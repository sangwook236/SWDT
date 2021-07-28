//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_torch {

void training_example();
void torch_script_example();

}  // namespace my_torch

int torch_main(int argc, char *argv[])
{
	std::cout << "torch library: Training example -----------------------------" << std::endl;
	my_torch::training_example();

	std::cout << "\ttorch library: Torch Script example -------------------------" << std::endl;
	my_torch::torch_script_example();

	return 0;
}
