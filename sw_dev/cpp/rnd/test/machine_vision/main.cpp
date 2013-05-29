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
	int opencv_main(int argc, char *argv[]);
	int vlfeat_main(int argc, char *argv[]);
	int ccv_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		// OpenCV library -------------------------------------------------
		retval = opencv_main(argc, argv);

		// VLFeat library -------------------------------------------------
		//retval = vlfeat_main(argc, argv);

		// CCV library ----------------------------------------------------
		//retval = ccv_main(argc, argv);  // run-time error (Windows): not correctly working
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
