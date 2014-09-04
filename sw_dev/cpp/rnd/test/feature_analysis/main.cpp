//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


// for compiling using NVIDIA NVCC CUDA Compiler.
//  [ref] http://benbarsdell.blogspot.kr/2009/03/cuda-in-codeblocks-first-things-second.html.

// for running SiftGPU & SURFGPU.
//  -. nvcc compiler가 path에 설정되어 있어야 함.
//      e.g.) export PATH=$PATH:/usr/local/cuda/bin

int main(int argc, char *argv[])
{
	int lsd_main(int argc, char *argv[]);
	int elsd_main(int argc, char *argv[]);
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

		std::cout << "line feature --------------------------------------------------------" << std::endl;
		retval = lsd_main(argc, argv);

		std::cout << "\nellipse & line feature ----------------------------------------------" << std::endl;
		//retval = elsd_main(argc, argv);

		std::cout << "\nlocal descriptors ---------------------------------------------------" << std::endl;
		//retval = opensift_main(argc, argv);
		//retval = siftgpu_main(argc, argv);

		//retval = opensurf_main(argc, argv);
		//retval = surfgpu_main(argc, argv);

		//retval = hog_main(argc, argv);

		std::cout << "\nFeature Selection Toolbox (FST) library -----------------------------" << std::endl;
		//retval = fst_main(argc, argv);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
