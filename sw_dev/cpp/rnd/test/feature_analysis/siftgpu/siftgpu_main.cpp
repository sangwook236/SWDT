//#include "stdafx.h"
#include <iostream>


// for building by NVCC
//  export PATH=$PATH:/usr/local/cuda/bin
//  export LD_LIBRARY_PATH+=/usr/local/MATLAB/R2012b/bin/glnxa64

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
