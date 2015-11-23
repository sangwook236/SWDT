//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#define GLOG_NO_ABBREVIATED_SEVERITIES 1
#include <glog/logging.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int glpk_main(int argc, char *argv[]);

	int ceres_solver_main(int argc, char *argv[]);
	int levmar_main(int argc, char *argv[]);

	int nlopt_main(int argc, char *argv[]);
	int optpp_main(int argc, char *argv[]);

	int galib_main(int argc, char *argv[]);

	int coin_or_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));
        google::InitGoogleLogging(argv[0]);

        std::cout << "GNU Linear Programming Kit (GLPK) library ---------------------------" << std::endl;
		//retval = glpk_main(argc, argv);

        std::cout << "\nCeres Solver --------------------------------------------------------" << std::endl;
        //  -. Non-linear least squares.
        //  -. General unconstrained minimization.
        //      Curve fitting.
        //      Robust curve fitting.
        //      Bundle adjustment.
		retval = ceres_solver_main(argc, argv);

		std::cout << "\nLevenberg-Marquardt (LM) algorithm ----------------------------------" << std::endl;
		//retval = levmar_main(argc, argv);

        std::cout << "\nNLopt library -------------------------------------------------------" << std::endl;
		//retval = nlopt_main(argc, argv);

        std::cout << "\nOPT++ library -------------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = optpp_main(argc, argv);
#else
		std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nGAlib library -------------------------------------------------------" << std::endl;
		//  -. Genetic algorithm.
		//retval = galib_main(argc, argv);

		std::cout << "\nComputational Infrastructure for Operations Research (COIN-OR) ------" << std::endl;
		//retval = coin_or_main(argc, argv);
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
