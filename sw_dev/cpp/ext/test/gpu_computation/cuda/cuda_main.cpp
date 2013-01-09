#include <cuda_runtime.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

}  // namespace my_cuda

#if defined(__cplusplus)
extern "C" {
#endif

//__global__ void HelloWorld2();
__global__ void HelloWorld2()
{
	std::cout << "Hello World!" << std::endl;
}

int cuda_main(int argc, char **argv)
{
	HelloWorld2();

    return 0;
}

#if defined(__cplusplus)
}  // extern "C"
#endif
