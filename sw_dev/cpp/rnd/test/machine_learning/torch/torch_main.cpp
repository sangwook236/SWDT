//include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_torch {

void deep_learning_example();

}  // namespace my_torch

int torch_main(int argc, char *argv[])
{
	std::cout << "torch library: deep learning example ----------------------------" << std::endl;
	my_torch::deep_learning_example();  // not yet implemented

	return 0;
}
