//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_fann {

void xor_example();

}  // namespace my_fann

int fann_main(int argc, char *argv[])
{
	// syncronize std::cout and printf output
	std::ios::sync_with_stdio();

	my_fann::xor_example();

	return 0;
}
