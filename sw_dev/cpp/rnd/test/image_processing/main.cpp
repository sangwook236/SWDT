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
	int itk_main(int argc, char *argv[]);
	int ritk_main(int argc, char *argv[]);
	int graphicsmagick_main(int argc, char *argv[]);
	int gegl_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::cout << "Insight Segmentation and Registration Toolkit (ITK) -----------------" << std::endl;
		retval = itk_main(argc, argv);  // Not yet implemented.

		std::cout << "\nThe Range Imaging Toolkit (RITK) ------------------------------------" << std::endl;
		retval = ritk_main(argc, argv);  // Not yet implemented.

		std::cout << "\nGraphicsMagick Image Processing System ------------------------------" << std::endl;
		retval = graphicsmagick_main(argc, argv);  // Not yet implemented.

		std::cout << "\nGeneric Graphics Library (GEGL) -------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		retval = gegl_main(argc, argv);  // Not yet implemented.
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif
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
