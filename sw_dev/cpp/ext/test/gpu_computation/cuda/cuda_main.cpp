#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void hello_world();
void basic_operation();

}  // namespace my_cuda

int cuda_main(int argc, char *argv[])
{
	//my_cuda::hello_world();

	my_cuda::basic_operation();

    return 0;
}
