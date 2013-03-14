//#include "stdafx.h"
#include <iostream>


// for building by NVCC
//  export PATH=$PATH:/usr/local/cuda/bin
//  export LD_LIBRARY_PATH+=/usr/local/MATLAB/R2012b/bin/glnxa64

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_surfgpu {

void example();

}  // namespace my_surfgpu

int surfgpu_main(int argc, char *argv[])
{
	my_surfgpu::example();

	return 0;
}
