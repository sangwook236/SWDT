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

void HelloWorld();

#if defined(__cplusplus)
}  // extern "C"
#endif

int cuda_main(int argc, char *argv[])
{
	HelloWorld();

    return 0;
}
