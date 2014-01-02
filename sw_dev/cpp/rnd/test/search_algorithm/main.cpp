//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int kdtree_main(int argc, char *argv[]);
	int libkdtreepp_main(int argc, char *argv[]);
	int ann_main(int argc, char *argv[]);
	int lshkit_main(int argc, char *argv[]);
	int slash_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "kdtree library ------------------------------------------------------" << std::endl;
		//retval = kdtree_main(argc, argv);

		std::cout << "\nlibkdtree++ library -------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = libkdtreepp_main(argc, argv);
#endif

		std::cout << "\nA Library for Approximate Nearest Neighbor Searching (ANN) ----------" << std::endl;
		retval = ann_main(argc, argv);

		std::cout << "\nLSHKIT library ------------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = lshkit_main(argc, argv);
#endif

		std::cout << "\nslash library -------------------------------------------------------" << std::endl;
		//  -. Spherical LSH (SLSH).
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = slash_main(argc, argv);
#endif
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
