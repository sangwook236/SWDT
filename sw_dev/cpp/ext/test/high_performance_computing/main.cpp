//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int openmp_main(int argc, char *argv[]);
	int simd_main(int argc, char *argv[]);

	int cuda_main(int argc, char *argv[]);
	int tensorrt_main(int argc, char *argv[]);
	int vexcl_main(int argc, char *argv[]);
	int thrust_main(int argc, char *argv[]);
	int arrayfire_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Open Multi-Processing (OpenMP) --------------------------------------" << std::endl;
		retval = openmp_main(argc, argv);

		std::cout << "\nSingle instruction, mutiple data (SIMD) -----------------------------" << std::endl;
		//	- Streaming SIMD Extensions (SSE).
		//retval = simd_main(argc, argv);

		std::cout << "\nBoost.MPI library ---------------------------------------------------" << std::endl;
		//	- Message Passing Interface (MPI).
		// REF [library] >> Boost library.

		std::cout << "\nBoost.Compute library -----------------------------------------------" << std::endl;
		//	- Open Computing Language (OpenCL).
		// REF [library] >> Boost library.

		std::cout << "\nCompute Unified Device Architecture (CUDA) --------------------------" << std::endl;
		//retval = cuda_main(argc, argv);

		std::cout << "\nTensorRT library ----------------------------------------------------" << std::endl;
		//retval = tensorrt_main(argc, argv);

		std::cout << "\nVexCL library -------------------------------------------------------" << std::endl;
		//	- Support OpenCL, CUDA, and Boost.Compute.
		//retval = vexcl_main(argc, argv);

		std::cout << "\nThrust library ------------------------------------------------------" << std::endl;
		//	- Interoperability with CUDA, TBB, and OpenMP.
		//retval = thrust_main(argc, argv);

		std::cout << "\nArrayFire library ---------------------------------------------------" << std::endl;
		//	- Support OpenCL, CUDA, and CPU.
		//retval = arrayfire_main(argc, argv);
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
