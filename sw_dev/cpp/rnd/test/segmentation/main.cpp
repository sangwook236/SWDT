//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


// for compiling using NVIDIA NVCC CUDA Compiler
//  [ref] http://benbarsdell.blogspot.kr/2009/03/cuda-in-codeblocks-first-things-second.html

int main(int argc, char *argv[])
{
	int region_growing_main(int argc, char *argv[]);
	int efficient_graph_based_image_segmentation_main(int argc, char *argv[]);
	int interactive_graph_cuts_main(int argc, char *argv[]);
	int gslic_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "region growing algorithm --------------------------------------------" << std::endl;
		//retval = region_growing_main(argc, argv);

		std::cout << "\nefficient graph-based image segmentation algorithm ------------------" << std::endl;
		//retval = efficient_graph_based_image_segmentation_main(argc, argv);

		std::cout << "\ninteractive graph-cuts algorithm ------------------------------------" << std::endl;
		// [ref] "Interactive Graph Cuts for Optimal Boundary & Region Segmentation of Objects in N-D Images".
		//retval = interactive_graph_cuts_main(argc, argv);

		std::cout << "\nsimple linear iterative clustering (SLIC) algorithm -----------------" << std::endl;
		retval = gslic_main(argc, argv);
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
		std::cerr << "unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
