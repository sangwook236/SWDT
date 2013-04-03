//#include "stdafx.h"
#if defined(WIN32)
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
	int maxflow_main(int argc, char *argv[]);
	int ibfs_main(int argc, char *argv[]);

	int graphlab_main(int argc, char *argv[]);
	int graphchi_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		// min-cut/max-flow algorithms ------------------------------
		retval = maxflow_main(argc, argv);  // not yet implemented
		retval = ibfs_main(argc, argv);  // not yet implemented

		// GraphLab library -----------------------------------------
		retval = graphlab_main(argc, argv);  // not yet implemented

		// GraphChi library -----------------------------------------
		retval = graphchi_main(argc, argv);  // not yet implemented
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
