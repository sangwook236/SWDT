//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


//-----------------------------------------------------------------------
// Basis spline (B-spline).
//	REF [site] >> https://en.wikipedia.org/wiki/B-spline
//	- Any spline function of given degree can be expressed as a linear combination of B-splines of that degree.
//	- Cardinal B-splines have knots that are equidistant from each other.
//	- B-splines can be used for curve-fitting and numerical differentiation of experimental data.

int main(int argc, char *argv[])
{
	int cgal_main(int argc, char *argv[]);
	int openmesh_main(int argc, char *argv[]);

	int sophus_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "Boost.Geometry & Boost.Polygon libraries -----------------------------" << std::endl;
		//	REF [library] >> Boost library.

		std::cout << "\nComputational Geometry Algorithms Library (CGAL) --------------------" << std::endl;
		retval = cgal_main(argc, argv);

		std::cout << "\nOpenMesh library ----------------------------------------------------" << std::endl;
		//retval = openmesh_main(argc, argv);

		std::cout << "\nSophus library ------------------------------------------------------" << std::endl;
		//	- Lie Groups.
		//retval = sophus_main(argc, argv);  // Not yet implemented.
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
