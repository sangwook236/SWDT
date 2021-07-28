//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_torch {

void deep_learning_example();
void torch_script_example();

}  // namespace my_torch

int torch_main(int argc, char *argv[])
{
	std::cout << "torch library: deep learning example ----------------------------" << std::endl;
	my_torch::deep_learning_example();  // Not yet implemented.

	std::cout << "\ntorch library: Torch Script example -----------------------------" << std::endl;
	my_torch::torch_script_example();

	return 0;
}
