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

		std::cout << "Boost library -------------------------------------------------------" << std::endl;
		//	- Algorithm.
		//		Boost.Algorithm.
		//		Boost.Sort.
		//		Boost.Min-Max.
		//	- String algorithm.
		//		Boost Tokenizer.
		//		Boost String Algorithms.
		//		Boost Format.
		//	- Regular expression.
		//		Boost.Regex.
		//		Boost.Xpressive.
		//	- Conversion.
		//		Boost.NumericConversion.
		//		Boost.Convert.
		//		Boost.Lexical_Cast.
		//		Boost Conversion Library: polymorphic cast.
		//	- State machine.
		//		Boost.Statechart.
		//		Boost Meta State Machine (MSM).
		//	- Parser.
		//		Boost Spirit.
		//		Boost.Metaparse.
		//		Boost.Proto.
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

		std::cout << "\nApproximate Nearest Neighbor (ANN) library --------------------------" << std::endl;
		//	- Approximate nearest neighbor search.
		retval = ann_main(argc, argv);

		std::cout << "\nFast Library for Approximate Nearest Neighbors (FLANN) --------------" << std::endl;
		//	- Approximate nearest neighbor search.
		//retval = flann_main(argc, argv);  // Not yet implemented.

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
