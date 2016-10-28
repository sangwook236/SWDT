//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
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

		std::cout << "Boost.Sort library --------------------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nkdtree library ------------------------------------------------------" << std::endl;
		//	- k-dimensional tree (k-d tree).
		//retval = kdtree_main(argc, argv);

		std::cout << "\nlibkdtree++ library -------------------------------------------------" << std::endl;
		//	- k-dimensional tree (k-d tree).
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = libkdtreepp_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nA Library for Approximate Nearest Neighbor Searching (ANN) ----------" << std::endl;
		//	- Approximate nearest neighbor.
		retval = ann_main(argc, argv);

		std::cout << "\nLSHKIT library ------------------------------------------------------" << std::endl;
		//	- Locality-sensitive hashing (LSH).
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = lshkit_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nslash library -------------------------------------------------------" << std::endl;
		//  - Spherical locality-sensitive hashing (SLSH).
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = slash_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
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
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
