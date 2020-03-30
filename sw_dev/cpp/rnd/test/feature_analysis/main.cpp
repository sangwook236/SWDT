//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


// For compiling using NVIDIA NVCC CUDA Compiler.
//  REF [site] >> http://benbarsdell.blogspot.kr/2009/03/cuda-in-codeblocks-first-things-second.html.

// For running SiftGPU & SURFGPU.
//  - Add the path of NVCC compiler to PATH.
//      e.g.) export PATH=$PATH:/usr/local/cuda/bin

int main(int argc, char *argv[])
{
	int opensift_main(int argc, char *argv[]);
	int siftgpu_main(int argc, char *argv[]);
	int opensurf_main(int argc, char *argv[]);
	int surfgpu_main(int argc, char *argv[]);
	int hog_main(int argc, char *argv[]);
	int fst_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Local descriptors ---------------------------------------------------" << std::endl;
		//retval = opensift_main(argc, argv);
		//retval = siftgpu_main(argc, argv);

		//retval = opensurf_main(argc, argv);
		//retval = surfgpu_main(argc, argv);

		//retval = hog_main(argc, argv);

		std::cout << "\nFeature Selection Toolbox (FST) library -----------------------------" << std::endl;
		//retval = fst_main(argc, argv);
	}
    catch (const std::bad_alloc &ex)
	{
		std::cout << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cout << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
