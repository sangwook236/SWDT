#include <iostream>
#include <cuda_runtime.h>
//#include <cutil.h>  // CUDA utility tools


#if defined(__CUDACC__)  // Defined only in .cu files
#error __CUDACC__ defined
#endif
#if defined(__CUDA_ARCH__)
#error __CUDA_ARCH__ not defined
#endif

namespace {
namespace local {

void basic_functionality()
{
	{
		int runtimeVersion = 0;
		cudaRuntimeGetVersion(&runtimeVersion);
		int driverVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		std::cout << "CUDA runtime version = " << runtimeVersion << ", CUDA driver version = " << driverVersion << std::endl;
	}

	// Error handling
	{
		std::cout << "Error handling:" << std::endl;

		std::cout << "\tError: " << cudaErrorNoDevice << std::endl;
		std::cout << "\tError name: " << cudaGetErrorName(cudaErrorNoDevice) << std::endl;
		std::cout << "\tError string: " << cudaGetErrorString(cudaErrorNoDevice) << std::endl;

		const auto lastErr = cudaGetLastError();
		//const auto lastErr = cudaPeekAtLastError();
		std::cout << "\tLast error: " << lastErr << std::endl;
		std::cout << "\tLast error name: " << cudaGetErrorName(lastErr) << std::endl;
		std::cout << "\tLast error string: " << cudaGetErrorString(lastErr) << std::endl;
	}

	// Event management
	{
		std::cout << "Event management:" << std::endl;

		// Timer
#if 0
		unsigned int timer;
		cutCreateTimer(&timer);
		cutStartTimer(timer)

		//kernel<<65535, 512>>(...);
		//cudaDeviceSynchronize();

		cutStopTimer(timer)
		const double elapsed_time = cutGetTimerValue(timer);
		std::cout << "\tElapsed time = " << elapsed_time << " msec." << std::endl;
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
		std::cout << "\tElapsed time = " << elapsed_time << " msec." << std::endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
#endif
	}

	// Device management
	{
		int device_count = -1;
		auto cudaStatus = cudaGetDeviceCount(&device_count);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaGetDeviceCount() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
		std::cout << "#devices = " << device_count << std::endl;

#if 0
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaSetDevice() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
#endif

		// The device on which the active host thread executes the device code
		int device = -1;
		cudaStatus = cudaGetDevice(&device);
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaGetDevice() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
		std::cout << "Current device = " << device << std::endl;

		cudaDeviceProp prop;
		cudaStatus = cudaGetDeviceProperties(&prop, device);
		if (cudaSuccess != cudaStatus)
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
		std::cout << "\tMax threads per block = " << prop.maxThreadsPerBlock << std::endl;
		std::cout << "\tMax threads dimension = " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << std::endl;
		std::cout << "\tMax gride size = " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
		std::cout << "\tMajor version = " << prop.major << std::endl;
		std::cout << "\tMinor version = " << prop.minor << std::endl;
		std::cout << "\tClock rate = " << prop.clockRate << std::endl;
		std::cout << "\t#SMs = " << prop.multiProcessorCount << std::endl;  // #SMs
		std::cout << "\tCan map host memory? = " << prop.canMapHostMemory << std::endl;
		std::cout << "\tCompute mode = " << prop.computeMode << std::endl;
		std::cout << "\tConcurrent kernels = " << prop.concurrentKernels << std::endl;

#if 0
		// cudaDeviceReset() must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaSuccess != cudaStatus)
		{
			std::cerr << "cudaDeviceReset() failed: " << cudaGetErrorString(cudaStatus) << std::endl;
			return;
		}
#endif
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_cuda {

void basic_operation();
void texture_test();

}  // namespace my_cuda

int cuda_main(int argc, char *argv[])
{
	local::basic_functionality();

	//-----
	my_cuda::basic_operation();

	//my_cuda::texture_test();

	return 0;
}
