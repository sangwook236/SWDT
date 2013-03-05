//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int lsd_main(int argc, char *argv[]);
	int opensift_main(int argc, char *argv[]);
	int siftgpu_main(int argc, char *argv[]);
	int opensurf_main(int argc, char *argv[]);
	int pictorial_structures_revisited_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		// line feature -----------------------------------------
		//retval = lsd_main(argc, argv);

		// local descriptor -------------------------------------
		//retval = opensift_main(argc, argv);
		retval = siftgpu_main(argc, argv);

		//retval = opensurf_main(argc, argv);
		
		// pictorial structures ---------------------------------
		//retval = pictorial_structures_revisited_main(argc, argv);
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
