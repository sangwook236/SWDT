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
	int fast_bilateral_filter_main(int argc, char *argv[]);
	int nyu_depth_toolbox_v2_main(int argc, char *argv[]);
	int itpp_main(int argc, char *argv[]);
	int spuc_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "fast bilateral filter algorithm -------------------------------------" << std::endl;
		//retval = fast_bilateral_filter_main(argc, argv);

		std::cout << "\nNYU Depth Toolbox V2 ------------------------------------------------" << std::endl;
		//retval = nyu_depth_toolbox_v2_main(argc, argv);

		std::cout << "\nIT++ library --------------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = itpp_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nSignal Processing using C++ (SPUC) library --------------------------" << std::endl;
		retval = spuc_main(argc, argv);
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
