#include <iostream>
#include <cuda_runtime_api.h>


#if defined(__CUDACC__)  // Defined only in .cu files
#error __CUDACC__ defined
#endif
#if defined(__CUDA_ARCH__)
#error __CUDA_ARCH__ not defined
#endif

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void basic_operation();
void texture_test();

}  // namespace my_cuda

int cuda_main(int argc, char *argv[])
{
	int runtimeVersion = 0;
	cudaRuntimeGetVersion(&runtimeVersion);
	int driverVersion = 0;
	cudaDriverGetVersion(&driverVersion);
	std::cout << "CUDA runtime version = " << runtimeVersion << ", CUDA driver version = " << driverVersion << std::endl;

	//-----
	my_cuda::basic_operation();

	//my_cuda::texture_test();

	return 0;
}
