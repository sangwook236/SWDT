//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int ransac_main(int argc, char *argv[]);
	int jlinkage_main(int argc, char *argv[]);

	int smctc_main(int argc, char *argv[]);
	int mcmcpp_main(int argc, char *argv[]);

	int scythe_main(int argc, char *argv[]);
	int boom_main(int argc, char *argv[]);

	int movmf_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)time(NULL));

		std::cout << "RANSAC algorithm ----------------------------------------------------" << std::endl;
		//	-. robust estimation.
		//retval = ransac_main(argc, argv);

		std::cout << "J-Linkage algorithm ----------------------------------------------------" << std::endl;
		//	-. robust multiple structures estimation.
		retval = jlinkage_main(argc, argv);

		std::cout << "\nSequential Monte Carlo Template Class (SMCTC) library --------------" << std::endl;
		//	-. sequential importance resampling (SIR) algorithm.
		//	-. particle filter.
		//	-. SMC sampler.
		//retval = smctc_main(argc, argv);

		std::cout << "\nMCMC++ library ------------------------------------------------------" << std::endl;
		//	-. Markov Chain Monte Carlo (MCMC) analysis.
		//retval = mcmcpp_main(argc, argv);

		std::cout << "\nScythe Statistical Library ------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = scythe_main(argc, argv);
#else
		std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nboom library --------------------------------------------------------" << std::endl;
		// Bayesian computation in C++.
		//retval = boom_main(argc, argv);  // not yet implemented.

		std::cout << "\nMoVMF library -------------------------------------------------------" << std::endl;
		//	-. directional statistics.
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = movmf_main(argc, argv);
#else
		std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif
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
