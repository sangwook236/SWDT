#include <arrayfire.h>
#include <iostream>


namespace {
namespace local {

// REF [site] >> http://arrayfire.org/docs/unifiedbackend.htm
void backend_test()
{
	const int device = 0;

	// Link with libafcpu.so or libaf.so (the unified backend).
	try
	{
		std::cout << "Trying ArrayFire CPU Backend." << std::endl;
		af::setBackend(AF_BACKEND_CPU);
		af::setDevice(device);
		af::info();

		af_print(af::randu(5, 4));
	}
	catch (const af::exception &ex)
	{
		std::cout << "Caught af::exception when trying ArrayFire CPU backend: " << ex.what() << std::endl;
	}

	// Link with libafcuda.so or libaf.so (the unified backend).
	try
	{
		std::cout << "Trying ArrayFire CUDA Backend." << std::endl;
		af::setBackend(AF_BACKEND_CUDA);
		af::setDevice(device);
		af::info();

		af_print(af::randu(5, 4));
	}
	catch (const af::exception &ex)
	{
		std::cout << "Caught af::exception when trying ArrayFire CUDA backend: " << ex.what() << std::endl;
	}

	// Link with libafopencl.so or libaf.so (the unified backend).
	try
	{
		std::cout << "Trying ArrayFire OpenCL Backend." << std::endl;
		af::setBackend(AF_BACKEND_OPENCL);
		af::setDevice(device);
		af::info();

		af_print(af::randu(5, 4));
	}
	catch (const af::exception &ex)
	{
		std::cout << "Caught af::exception when trying ArrayFire OpenCL backend: " << ex.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_arrayfire {

void morphology();

}  // namespace my_arrayfire

int arrayfire_main(int argc, char *argv[])
{
	try
	{
		local::backend_test();

		my_arrayfire::morphology();
	}
	catch (const af::exception &ex)
	{
		std::cout << "af::exception caught: " << ex.what() << std::endl;

		return 1;
	}

	return 0;
}
