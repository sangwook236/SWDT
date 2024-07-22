//include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int libpabod_main(int argc, char *argv[]);
	int object_detection_and_localization_toolkit_main(int argc, char *argv[]);
	int object_detection_toolbox_main(int argc, char *argv[]);

	int c4_main(int argc, char *argv[]);

	int shadows_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "a LIBrary for PArt-Based Object Detection in C++ (LibPaBOD) ---------" << std::endl;
		//	- Discriminatively trained part-based model.
		//retval = libpabod_main(argc, argv);

		std::cout << "\nINRIA Object Detection and Localization Toolkit ---------------------" << std::endl;
		//	- Histogram of oriented gradients (HOG).
		//retval = object_detection_and_localization_toolkit_main(argc, argv);  // Not yet implemented.

		std::cout << "\nObject Detection Toolbox --------------------------------------------" << std::endl;
		//	- Structured SVM.
		//retval = object_detection_toolbox_main(argc, argv);  // Not yet implemented.

		std::cout << "\nC4 detector ---------------------------------------------------------" << std::endl;
		// C4: Real-time pedestrian detection.
		//	- CENTRIST descriptor.
		//retval = c4_main(argc, argv);

		std::cout << "\nOpenCV Saliency API -------------------------------------------------" << std::endl;
		// REF [library] >> OpenCV Saliency API.

		std::cout << "\nShadow detection and removal algorithm ------------------------------" << std::endl;
		retval = shadows_main(argc, argv);
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
