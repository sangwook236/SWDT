//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int openahrs_main(int argc, char *argv[]);
	int open_source_ahrs_main(int argc, char *argv[]);
	int open_robotics_ars_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Open AHRS (Attitude and Heading Reference System) library ----------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		retval = openahrs_main(argc, argv);  // Not yet implemented.
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nOpen source IMU and AHRS algorithms --------------------------------" << std::endl;
		//retval = open_source_ahrs_main(argc, argv);

		std::cout << "\nOpen Robotics ARS algorithms ---------------------------------------" << std::endl;
		retval = open_robotics_ars_main(argc, argv);
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
