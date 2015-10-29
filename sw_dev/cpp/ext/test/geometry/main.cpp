//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int sophus_main(int argc, char *argv[]);

	int cgal_main(int argc, char *argv[]);
	int openmesh_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Sophus library -------------------------------------------------------" << std::endl;
		//	-. Lie Groups.
		//retval = sophus_main(argc, argv);  // not yet implemented.

		std::cout << "\nComputational Geometry Algorithms Library (CGAl) --------------------" << std::endl;
		retval = cgal_main(argc, argv);

		std::cout << "\nOpenMesh library ----------------------------------------------------" << std::endl;
		retval = openmesh_main(argc, argv);
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
