//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

int main(int argc, char *argv[])
{
	int astar_algorithm_cpp_main(int argc, char *argv[]);

	int maxflow_main(int argc, char *argv[]);
	int ibfs_main(int argc, char *argv[]);

	int graphlab_main(int argc, char *argv[]);
	int graphchi_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "\nastar-algorithm-cpp algorithm ---------------------------------------" << std::endl;
		//retval = astar_algorithm_cpp_main(argc, argv);  // Not yet implemented.

		std::cout << "\nmin-cut/max-flow algorithms -----------------------------------------" << std::endl;
		retval = maxflow_main(argc, argv);  // Not yet implemented.
		retval = ibfs_main(argc, argv);  // Not yet implemented.

		std::cout << "\nGraphLab library ----------------------------------------------------" << std::endl;
		retval = graphlab_main(argc, argv);  // Not yet implemented.

		std::cout << "\nGraphChi library ----------------------------------------------------" << std::endl;
		retval = graphchi_main(argc, argv);  // Not yet implemented.
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
