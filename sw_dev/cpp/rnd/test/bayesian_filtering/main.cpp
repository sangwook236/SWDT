//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int bayespp_main(int argc, char *argv[]);
	int particleplusplus_main(int argc, char *argv[]);
	int condensation_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Bayes++ library ----------------------------------------------------" << std::endl;
		//	-. Kalman filter (KF).
		//	-. extended Kalman filter (EKF).
		//	-. unscented Kalman filter (UKF).
		//	-. covariance intersection filter.
		//	-. U*d*U' factorisation of covariance filter.
        //      a 'Square-root' linearised Kalman filter.
		//	-. covariance filter.
		//	-. information filter.
		//	-. information root filter.
		//	-. iterated covariance filter.
		//	-. sampling importance resampling (SIR) filter.
		//      particle filter.
		//  -. simultaneous localization and mapping (SLAM).
		retval = bayespp_main(argc, argv);

		std::cout << "\nParticle++ library -------------------------------------------------" << std::endl;
		//	-. particle filter.
		//	-. sequential Monte Carlo (SMC) method.
		//retval = particleplusplus_main(argc, argv);

		std::cout << "\nCONDENATION (CONditional DENsity propagATION) algorithm -----------" << std::endl;
		//retval = condensation_main(argc, argv);
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
