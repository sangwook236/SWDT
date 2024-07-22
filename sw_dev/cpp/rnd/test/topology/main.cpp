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
	int dynamic_time_warping_main(int argc, char *argv[]);
	int fastemd_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Dynamic time warping (DTW) algorithm --------------------------------" << std::endl;
		retval = dynamic_time_warping_main(argc, argv);

		std::cout << "\nFastEMD algorithm ---------------------------------------------------" << std::endl;
		//	- Earth mover's distance (EMD).
		//retval = fastemd_main(argc, argv);  // Not yet implemented.
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
