//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_siftgpu {

void simple_example_1();
void simple_example_2();

}  // namespace my_siftgpu

int siftgpu_main(int argc, char *argv[])
{
	//my_siftgpu::simple_example_1();
	my_siftgpu::simple_example_2();

	return 0;
}
