//include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int libsvm_main(int argc, char *argv[]);
	int mysvm_main(int argc, char *argv[]);
	int clustering_main(int argc, char *argv[]);
	int vlfeat_main(int argc, char *argv[]);
	int rl_glue_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)time(NULL));

		// SVM --------------------------------------------------
		//retval = libsvm_main(argc, argv);  // not yet implemented
		//retval = mysvm_main(argc, argv);  // not yet implemented

		// clustering -------------------------------------------
		//retval = clustering_main(argc, argv);  // not yet implemented
		//retval = vlfeat_main(argc, argv);  // not yet implemented

		// reinforcement learning -------------------------------
		//retval = rl_glue_main(argc, argv);  // not yet implemented
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
