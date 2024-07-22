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
	int particle_filter_object_tracking_main(int argc, char *argv[]);
	int klt_main(int argc, char *argv[]);
	int mht_main(int argc, char *argv[]);
	int opentld_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Particle filter object tracking algorithm --------------------------" << std::endl;
		retval = particle_filter_object_tracking_main(argc, argv);

		std::cout << "\nKanade-Lucas-Tomasi (KLT) Feature Tracker algorithm -----------------" << std::endl;
		//retval = klt_main(argc, argv);

		std::cout << "\nMultiple Hypothesis Tracking (MHT) algorithm ------------------------" << std::endl;
		//retval = mht_main(argc, argv);  // Not yet implemented.

		std::cout << "\nOpenTLD algorithm ---------------------------------------------------" << std::endl;
		//retval = opentld_main(argc, argv);
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
