//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int adis_main(int argc, char *argv[]);
	int invensense_main(int argc, char *argv[]);
	int sparkfun_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Analog Devices MEMS Sensors (ADIS) ---------------------------------" << std::endl;
		retval = adis_main(argc, argv);

		std::cout << "\nInvenSense Inertial Measurement Unit (IMU) -------------------------" << std::endl;
		//retval = invensense_main(argc, argv);

		std::cout << "\nSparkfun Inertial Measurement Unit (IMU) ---------------------------" << std::endl;
		//retval = sparkfun_main(argc, argv);  // not yet implemented.
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
