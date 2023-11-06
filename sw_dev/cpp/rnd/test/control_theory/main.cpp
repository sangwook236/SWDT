//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <cstdlib>
#include <stdexcept>
#include <iostream>


int main(int argc, char *argv[])
{
	int control_toolbox_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Control Toolbox ----------------------------------------------------" << std::endl;
		retval = control_toolbox_main(argc, argv);  // Not yet implemented.

		std::cout << "\ndlib library --------------------------------------------------------" << std::endl;
		//	- Model predictive control (MPC).
		// REF [file] >> ${SWDT_C++_HOME}/ext/test/general_purpose_library/dlib/dlib_control_example.cpp.
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
