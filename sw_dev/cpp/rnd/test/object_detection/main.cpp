//include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int libpabod_main(int argc, char *argv[]);
	int object_detection_toolbox_main(int argc, char *argv[]);

	int c4_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "a LIBrary for PArt-Based Object Detection in C++ (LibPaBOD) ---------" << std::endl;
		//	-. discriminatively trained part-based model.
		//retval = libpabod_main(argc, argv);

		std::cout << "\nObject Detection Toolbox --------------------------------------------" << std::endl;
		//	-. structured SVM.
		//retval = object_detection_toolbox_main(argc, argv);  // not yet implemented.

		std::cout << "\nC4 detector ---------------------------------------------------------" << std::endl;
		retval = c4_main(argc, argv);
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
