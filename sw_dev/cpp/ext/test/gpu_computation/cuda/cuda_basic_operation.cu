#include <cuda_runtime.h>
#include <iostream>


namespace {
namespace local {

__global__ void simple_kernel_function(int a, int b, int c, int *sum)
{
	*sum = a + b + c;
}

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void hello_world()
{
	std::cout << "Hello World!" << std::endl;
}

void basic_operation()
{
	{
		const int input_data[5] = { 1, 2, 3, 4, 5 };
		int output_data[5] = { 0, };

		int *device_memory = NULL;

		// allocate memory in device
		cudaMalloc((void **)&device_memory, 5 * sizeof(int));

		// host -> device
		cudaMemcpy(device_memory, input_data, 5 * sizeof(int), cudaMemcpyHostToDevice);

		// device -> host
		cudaMemcpy(output_data, device_memory, 5 * sizeof(int), cudaMemcpyDeviceToHost);

		for (int i = 0; i < 5; ++i)
			std::cout << "output data[" << i << "] = " << output_data[i] << std::endl;

		cudaFree(device_memory);
	}

	{
		int *sum_in_device = NULL;
		cudaMalloc((void **)&sum_in_device, sizeof(int));

		local::simple_kernel_function<<<6,6>>>(1, 2, 3, sum_in_device);

#if 0
		// run-time error: access to device memory
		std::cout << "CUDA kernel function => sum = " << *sum_in_device << std::endl;
#else
		// device -> host
		// not correctly working
		int sum = 0;
		cudaMemcpy(sum_in_device, &sum, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "CUDA kernel function => sum = " << sum << std::endl;
#endif

		cudaFree(sum_in_device);
	}
}

}  // namespace my_cuda
