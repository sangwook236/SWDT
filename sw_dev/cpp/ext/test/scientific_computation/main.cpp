//#include "stdafx.h"
#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int gsl_main(int argc, char *argv[]);
	int gslwrap_main(int argc, char *argv[]);
	int alglib_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "GNU Scientific Library (GSL) ----------------------------------------" << std::endl;
		//retval = gsl_main(argc, argv);
		std::cout << "\nGSLwrap library -----------------------------------------------------" << std::endl;
		//retval = gslwrap_main(argc, argv);

		std::cout << "\nALGLIB library ------------------------------------------------------" << std::endl;
		//	- Linear algebra.
		//	- Interpolation.
		//		Spline: Linear splien, Hermite spline, Catmull-Rom spline, Cubic spline, Akima spline.
		//	- Differentiataion & integration
		//	- Optimization.
		//	- Statistics.
		retval = alglib_main(argc, argv);
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
