//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int condensation_main(int argc, char *argv[]);
	int particleplusplus_main(int argc, char *argv[]);
	int smctc_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "condensation (CONditional DENsity propagATION) algorithm -----------" << std::endl;
		retval = condensation_main(argc, argv);

		std::cout << "\nParticle++ library -------------------------------------------------" << std::endl;
		//retval = particleplusplus_main(argc, argv);

		std::cout << "\nSequential Monte Carlo Template Class (SMCTC) library --------------" << std::endl;
		//retval = smctc_main(argc, argv);
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
