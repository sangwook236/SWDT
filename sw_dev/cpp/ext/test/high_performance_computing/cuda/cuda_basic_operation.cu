#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
//#include <cutil.h>  // CUDA utility tools


#if defined(__CUDACC__)  // Defined only in .cu files
#define STR(x) #x
#define XSTR(x) STR(x)
#pragma message("#####>>>>> __CUDACC__ = " XSTR(__CUDACC__))
#pragma message("#####>>>>> __CUDACC_VER_MAJOR__ = " XSTR(__CUDACC_VER_MAJOR__))
#pragma message("#####>>>>> __CUDACC_VER_MINOR__ = " XSTR(__CUDACC_VER_MINOR__))
#pragma message("#####>>>>> __CUDACC_VER_BUILD__ = " XSTR(__CUDACC_VER_BUILD__))
#else
#error __CUDACC__ not defined
#endif

#if defined(__CUDA_ARCH__)
#define STR(x) #x
#define XSTR(x) STR(x)
#pragma message("#####>>>>> __CUDA_ARCH__ = " XSTR(__CUDA_ARCH__))
#else
//#error __CUDA_ARCH__ not defined
#endif

#if __CUDA_ARCH__ >= 800  // Device code path for compute capability 8.x
#elif __CUDA_ARCH__ >= 700  // Device code path for compute capability 7.x
#elif __CUDA_ARCH__ >= 600  // Device code path for compute capability 6.x
#elif __CUDA_ARCH__ >= 500  // Device code path for compute capability 5.x
#elif !defined(__CUDA_ARCH__)  // Host code path
#endif

namespace {
namespace local {

// REF [site] >> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//	Device-side symbols (i.e., those marked __device__) may be referenced from within a kernel simply via the & operator, as all global-scope device variables are in the kernel's visible address space.
//	This also applies to __constant__ symbols, although in this case the pointer will reference read-only data.

// NOTE [info] >> (global-scope) device variables can't be accessed on host
__constant__ int const_val_dev = 123;
//__constant__ int *const_ptr_dev = nullptr;  // NOTE [info] >> Bad practice
__constant__ int *const_ptr_dev = &const_val_dev;
__constant__ int const_arr_dev[1] = { -1, };
__device__ int val_dev;
__device__ int *ptr_dev = nullptr;
__device__ int arr_dev[8192] = { -1, };

__global__ void allocate_device_variables(size_t sz)
{
	//const_ptr_dev = (int *)malloc(sizeof(int));  // Compile-time error: expression must be a modifiable lvalue
	ptr_dev = (int *)malloc(sizeof(int) * sz);
}

__global__ void deallocate_device_variables()
{
	//free(const_ptr_dev);
	//const_ptr_dev = nullptr;  // Compile-time error: expression must be a modifiable lvalue
	free(ptr_dev);
	ptr_dev = nullptr;
}

__global__ void kernel_for_device_variables(int *p, size_t sz)
{
	//int ti = threadIdx.x;
	int ti = blockIdx.x * blockDim.x + threadIdx.x;
	//int tj = blockIdx.y * blockDim.y + threadIdx.y;
	//int tk = blockIdx.z * blockDim.z + threadIdx.z;

	//const_val_dev = 321;  // Compile-time error: expression must be a modifiable lvalue
	*const_ptr_dev = 654;  // Runtime error: an illegal memory access was encountered
	//const_ptr_dev = &const_val_dev;  // Compile-time error: expression must be a modifiable lvalue
	//const_arr_dev[0] = 987;  // Runtime error: unspecified launch failure
	val_dev = const_val_dev;
	val_dev *= 2;

#if 0
	// NOTE [info] >> It is recommended to use a kernel for memory allocation and deallocation

	if (0 == ti)
		ptr_dev = (int *)malloc(sizeof(int) * sz);

	// A runtime error, "an illegal memory access was encountered" occurs when trying to access ptr_dev, if threads aren't synced
	__syncthreads();
#endif

	if (ti < sz)
	{
		p[ti] = p[ti] * 4 + 1;
		ptr_dev[ti] = p[ti] * 4 + 2;  // A runtime error, "an illegal memory access was encountered" occurs when trying to access ptr_dev, if ptr_dev is not allocated before kernel_for_device_variables() is called
		//atomicAdd(ptr_dev + ti, ti);  // A runtime error, "an illegal memory access was encountered" occurs when trying to access ptr_dev, if ptr_dev is not allocated before kernel_for_device_variables() is called
		arr_dev[ti] = p[ti] * 4 + 3;
	}

	__syncthreads();  // Not necessary

	if (0 == ti)
		// A runtime error, "an illegal memory access was encountered" occurs when trying to access const_ptr_dev, even if const_ptr_dev is allocated before kernel_for_device_variables() is called
		printf("On device: const_val_dev = %d, *const_ptr_dev = %d, const_arr_dev[0] = %d, val_dev = %d, ptr_dev[0] = %d, arr_dev[0] = %d, p[0] = %d\n", const_val_dev, *const_ptr_dev, const_arr_dev[0], val_dev, ptr_dev[0], arr_dev[0], p[0]);
		//printf("On device: const_val_dev = %d, const_arr_dev[0] = %d, val_dev = %d, ptr_dev[0] = %d, arr_dev[0] = %d, p[0] = %d\n", const_val_dev, const_arr_dev[0], val_dev, ptr_dev[0], arr_dev[0], p[0]);

#if 0
	// NOTE [info] >> It is recommended to use a kernel for memory allocation and deallocation

	if (0 == ti)
	{
		free(ptr_dev);
		ptr_dev = nullptr;
	}
#endif
}

void access_device_variables()
{
	const size_t DAT_SIZE = 8192;
	const size_t NUM_THREADS_PER_BLOCK = 1024;
	const size_t NUM_BLOCKS = DAT_SIZE / NUM_THREADS_PER_BLOCK;

	std::cout << "#blocks = " << NUM_BLOCKS << ", #threads per block = " << NUM_THREADS_PER_BLOCK << std::endl;

	cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float elapsed_time = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	const int const_val_host = 369;
	int *p_h = nullptr;
	int *p_d = nullptr;

	p_h = new int [DAT_SIZE];
	for (size_t i = 0; i < DAT_SIZE; ++i)
		p_h[i] = int(i) + 1;
	//std::cout << "On host (before): p_h[0] = " << p_h[0] << std::endl;
	std::cout << "On host (before): ";
	//std::copy(p_h, p_h + DAT_SIZE, std::ostream_iterator<int>(std::cout, ", "));
	std::copy(p_h, p_h + 5, std::ostream_iterator<int>(std::cout, ", "));
	std::cout << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaMalloc(&p_d, sizeof(int) * DAT_SIZE);
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaMalloc(): " << elapsed_time << " msec." << std::endl;

	//cudaStatus = cudaMalloc(&const_ptr_dev, sizeof(int));
	//if (cudaSuccess != cudaStatus)
	//{
	//	std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	//	return;
	//}

	cudaEventRecord(start, 0);
	allocate_device_variables<<<1, 1>>>(DAT_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "allocate_device_variables() kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for allocate_device_variables(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaDeviceSynchronize();  // Blocks until the device has completed all preceding requested tasks.
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaDeviceSynchronize(): " << elapsed_time << " msec." << std::endl;

	//cudaStatus = cudaMemcpyToSymbol(&const_val_dev, &const_val_host, sizeof(int), 0, cudaMemcpyHostToDevice);  // Runtime error: invalid device symbol
	//if (cudaSuccess != cudaStatus)
	//{
	//	std::cerr << "cudaMemcpyToSymbol() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	//	return;
	//}
	//cudaStatus = cudaMemcpyToSymbol(const_ptr_dev, &const_val_host, sizeof(int), 0, cudaMemcpyHostToDevice);  // REF [function] >> kernel_for_device_variables()
	//if (cudaSuccess != cudaStatus)
	//{
	//	std::cerr << "cudaMemcpyToSymbol() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	//	return;
	//}
	cudaEventRecord(start, 0);
	cudaStatus = cudaMemcpyToSymbol(const_arr_dev, &const_val_host, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaMemcpyToSymbol() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaMemcpyToSymbol(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaMemcpy(p_d, p_h, sizeof(int) * DAT_SIZE, cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaMemcpy() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaMemcpy(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	//kernel_for_device_variables<<<DAT_SIZE, 1>>>(p_d, DAT_SIZE);
	kernel_for_device_variables<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(p_d, DAT_SIZE);
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "kernel_for_device_variables() kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for kernel_for_device_variables(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaDeviceSynchronize();  // Blocks until the device has completed all preceding requested tasks.
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaDeviceSynchronize(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	deallocate_device_variables<<<1, 1>>>();
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "deallocate_device_variables() kernel failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for deallocate_device_variables(): " << elapsed_time << " msec." << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaDeviceSynchronize();  // Blocks until the device has completed all preceding requested tasks.
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaDeviceSynchronize(): " << elapsed_time << " msec." << std::endl;

	//std::cout << "On host: *p_d = " << *p_d << std::endl;  // Runtime error: killed
	cudaEventRecord(start, 0);
	cudaStatus = cudaMemcpy(p_h, p_d, sizeof(int) * DAT_SIZE, cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaMemcpy() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaMemcpy(): " << elapsed_time << " msec." << std::endl;

	//std::cout << "On host (after): p_h[0] = " << p_h[0] << std::endl;
	std::cout << "On host (after): ";
	//std::copy(p_h, p_h + DAT_SIZE, std::ostream_iterator<int>(std::cout, ", "));
	std::copy(p_h, p_h + 5, std::ostream_iterator<int>(std::cout, ", "));
	std::cout << std::endl;

	cudaEventRecord(start, 0);
	cudaStatus = cudaFree(p_d);
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaFree() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
	p_d = nullptr;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	std::cout << "Elapsed time for cudaFree(): " << elapsed_time << " msec." << std::endl;

	udaEventRecord(start, 0);
	//cudaStatus = cudaFree(const_ptr_dev);
	//if (cudaSuccess != cudaStatus)
	//{
	//	std::cerr << "cudaFree() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
	//	return;
	//}
	//const_ptr_dev = nullptr;  // Compile-time error: expression must be a modifiable lvalue

	delete [] p_h;
	p_h = nullptr;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

__global__ void sum_kernel(int a, int b, int c, int *sum)
{
	*sum = a + b + c;
}

void simple_example_1()
{
	{
		const int input_data[5] = { 1, 2, 3, 4, 5 };
		int output_data[5] = { 0, };

		int *ptr_dev = nullptr;

		// Allocate memory in device
		cudaMalloc((void **)&ptr_dev, 5 * sizeof(int));

		// Host -> device
		cudaMemcpy(ptr_dev, input_data, 5 * sizeof(int), cudaMemcpyHostToDevice);

		// Device -> host
		cudaMemcpy(output_data, ptr_dev, 5 * sizeof(int), cudaMemcpyDeviceToHost);

		for (auto i = 0; i < 5; ++i)
			std::cout << "Output data[" << i << "] = " << output_data[i] << std::endl;

		cudaFree(ptr_dev);
	}

	{
		int *sum_dev = nullptr;
		cudaMalloc((void **)&sum_dev, sizeof(int));

		sum_kernel<<<6, 6>>>(1, 2, 3, sum_dev);  // <<<...>>>: execution configuration syntax
		cudaDeviceSynchronize();

		//std::cout << "Direct access of a device variable: sum = " << *sum_dev << std::endl;  // Runtime error: access to device memory

		// Device -> host.
		int sum = 0;
		std::cout << "Before calling CUDA kernel function: sum = " << sum << std::endl;
		cudaMemcpy(&sum, sum_dev, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "After calling CUDA kernel function: sum = " << sum << std::endl;

		cudaFree(sum_dev);
	}
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void **)&dev_c, size * sizeof(int));
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMalloc() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_a, size * sizeof(int));
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMalloc() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMalloc((void **)&dev_b, size * sizeof(int));
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMalloc() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "addKernel() kernel failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	
	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaDeviceSynchronize() returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "cudaMemcpy() failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	
	return cudaStatus;
}

void simple_example_2()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0, };
	cudaError_t cudaStatus;

	// Add vectors in parallel.
	cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaSuccess != cudaStatus)
	{
		fprintf(stderr, "addWithCuda() failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n", c[0], c[1], c[2], c[3], c[4]);
}

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void basic_operation()
{
	cudaError_t cudaStatus = cudaSuccess;

	// Error handling
	{
		std::cout << "Error: " << cudaErrorNoDevice << std::endl;
		std::cout << "Error name: " << cudaGetErrorName(cudaErrorNoDevice) << std::endl;
		std::cout << "Error string: " << cudaGetErrorString(cudaErrorNoDevice) << std::endl;

		const auto lastErr = cudaGetLastError();
		//const auto lastErr = cudaPeekAtLastError();
		std::cout << "Last error: " << lastErr << std::endl;
		std::cout << "Last error name: " << cudaGetErrorName(lastErr) << std::endl;
		std::cout << "Last error string: " << cudaGetErrorString(lastErr) << std::endl;
	}

	//-----
	// Device
	{
		int device_count = -1;
		cudaStatus = cudaGetDeviceCount(&device_count);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaGetDeviceCount() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
		std::cout << "#devices = " << device_count << std::endl;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaSetDevice() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}

		int device = -1;
		cudaStatus = cudaGetDevice(&device);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaGetDevice() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
		std::cout << "Device ID = " << device << std::endl;

		cudaDeviceProp prop;
		cudaStatus = cudaGetDeviceProperties(&prop, device);
		{
			std::cerr << "cudaGetDeviceProperties() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
		std::cout << "Device properties:" << std::endl;
		std::cout << "\tName: " << prop.name << std::endl;
		std::cout << "\tTotal global memory = " << prop.totalGlobalMem << std::endl;
		std::cout << "\tShared memory per block = " << prop.sharedMemPerBlock << std::endl;
		std::cout << "\t#registers(32bits) per block = " << prop.regsPerBlock << std::endl;
		std::cout << "\tWarp size = " << prop.warpSize << std::endl;
		std::cout << "\tmax threads per block = " << prop.maxThreadsPerBlock << std::endl;
		std::cout << "\tmax threads dimension = " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
		std::cout << "\tmax gride size = " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
		std::cout << "\tMajor version = " << prop.major << std::endl;
		std::cout << "\tMinor version = " << prop.minor << std::endl;
		std::cout << "\tClock rate = " << prop.clockRate << std::endl;
		std::cout << "\t#SMs = " << prop.multiProcessorCount << std::endl;  // #SMs
		std::cout << "\tCan map host memory = " << prop.canMapHostMemory << std::endl;
		std::cout << "\tCompute mode = " << prop.computeMode << std::endl;
		std::cout << "\tConcurrent kernels = " << prop.concurrentKernels << std::endl;
	}

	//-----
	// Timer
	{
#if 0
		unsigned int timer;
		cutCreateTimer(&timer);
		cutStartTimer(timer)

		//kernel<<65535, 512>>(...);
		//cudaDeviceSynchronize();

		cutStopTimer(timer)
		const double elapsed_time = cutGetTimerValue(timer);
		std::cout << "Elapsed time = " << elapsed_time << " msec." << std::endl;
#else
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);

		//kernel<<65535, 512>>(...);
		//cudaDeviceSynchronize();

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float elapsed_time;
		cudaEventElapsedTime(&elapsed_time, start, stop);
		//cudaEventElapsedTime_v2(&elapsed_time, start, stop);
		std::cout << "Elapsed time = " << elapsed_time << " msec." << std::endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
#endif
	}

	//-----
	local::access_device_variables();

	//local::simple_example_1();
	//local::simple_example_2();

	// cudaDeviceReset() must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaSuccess != cudaStatus)
	{
		std::cerr << "cudaDeviceReset() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
		return;
	}
}

}  // namespace my_cuda
