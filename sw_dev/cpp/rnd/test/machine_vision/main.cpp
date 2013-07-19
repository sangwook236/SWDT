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
	int vxl_main(int argc, char *argv[]);
	int ivt_main(int argc, char *argv[]);
	int vlfeat_main(int argc, char *argv[]);
	int ccv_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "OpenCV library ------------------------------------------------------" << std::endl;
		retval = opencv_main(argc, argv);

		std::cout << "\nVXL (the Vision-something-Libraries) library ------------------------" << std::endl;
		//retval = vxl_main(argc, argv);

		std::cout << "\nIntegrating Vision Toolkit (IVT) library ----------------------------" << std::endl;
		//retval = ivt_main(argc, argv);

		std::cout << "\nVLFeat library ------------------------------------------------------" << std::endl;
		//retval = vlfeat_main(argc, argv);

		std::cout << "\nCCV library ---------------------------------------------------------" << std::endl;
		//retval = ccv_main(argc, argv);  // run-time error in Windows: not correctly working.
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
