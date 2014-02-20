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
	int particle_filter_object_tracking_main(int argc, char *argv[]);
	int klt_main(int argc, char *argv[]);
	int opentld_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "particle filter object tracking algorithm --------------------------" << std::endl;
		retval = particle_filter_object_tracking_main(argc, argv);

		std::cout << "\nKanade-Lucas-Tomasi (KLT) Feature Tracker algorithm -----------------" << std::endl;
		//retval = klt_main(argc, argv);

		std::cout << "\nOpenTLD algorithm ---------------------------------------------------" << std::endl;
		//retval = opentld_main(argc, argv);
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
