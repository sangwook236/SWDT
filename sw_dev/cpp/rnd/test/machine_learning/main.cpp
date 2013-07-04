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
	int clustering_main(int argc, char *argv[]);

	int libsvm_main(int argc, char *argv[]);
	int mysvm_main(int argc, char *argv[]);

	int shogun_main(int argc, char *argv[]);

	int rl_glue_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)time(NULL));

		// clustering -------------------------------------------
		retval = clustering_main(argc, argv);

		// support vector machine (SVM) -------------------------
		//retval = libsvm_main(argc, argv);
		//retval = mysvm_main(argc, argv);  // not yet implemented

		// shogun library ---------------------------------------
		//	-. multiple kernel learning (MKL)
		//	-. Gaussian process (GP) regression
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		retval = shogun_main(argc, argv);
#endif

		// reinforcement learning (RL) --------------------------
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
