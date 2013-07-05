//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char* argv[])
{
	int levmar_main(int argc, char *argv[]);
	int galib_main(int argc, char *argv[]);

	int glpk_main(int argc, char *argv[]);
	int nlopt_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		// ------------------------------------------------
		//retval = levmar_main(argc, argv);

		// genetic algorithm ------------------------------
		//retval = galib_main(argc, argv);

		// GNU Linear Programming Kit (GLPK) --------------
		retval = glpk_main(argc, argv);

		// NLopt library ----------------------------------
		//retval = nlopt_main(argc, argv);
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
