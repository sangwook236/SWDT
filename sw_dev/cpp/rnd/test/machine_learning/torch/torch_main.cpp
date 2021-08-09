//include "stdafx.h"
#include <iostream>
#include <torch/torch.h>

// LibTorch:
//	Download from https://pytorch.org/


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
	try
	{
		std::cout << "torch library: Training example -----------------------------" << std::endl;
		my_torch::training_example();

		std::cout << "\ttorch library: Torch Script example -------------------------" << std::endl;
		my_torch::torch_script_example();
	}
	catch (const c10::Error &ex)
	{
		std::cout << "c10::Error caught: " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
