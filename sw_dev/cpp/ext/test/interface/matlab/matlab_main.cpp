//#include "stdafx.h"
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_matlab {

void matio();

}  // namespace my_matlab

int matlab_main(int argc, char *argv[])
{
	std::cout << "MAT File I/O Library (matio) ----------------------------------------" << std::endl;
	my_matlab::matio();  // Not yet implemented.

	return 0;
}
