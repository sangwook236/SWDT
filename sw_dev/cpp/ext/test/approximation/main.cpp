#if defined(_WIN64) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int tinyspline_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "GSL library -------------------------------------------------------" << std::endl;
		//	- Polynomial interpolation.
		//		Spline and its differentiation and integration.
		//	- Chebyshev approximation.
		//	- Linear/nonlinear least-squares fitting.
		//	- Basis spline (B-spline).
		// REF [library] >> GSL library.

		std::cout << "\nALGLIB library ------------------------------------------------------" << std::endl;
		//	- Interpolation and fitting.
		//		Polynomial interpolation.
		//		Rational interpolation.
		//		Spline interpolation.
		//			Spline and its differentiation and integration.
		//			Bilinear and bicubic spline interpolation.
		//		Linear/nonlinear least squares fitting.
		//		Fast RBF interpolation/fitting.
		// REF [library] >> ALGLIB library.

		std::cout << "\nEigen Spline library ------------------------------------------------" << std::endl;
		// REF [library] >> Eigen library.

		std::cout << "\nTinySpline library --------------------------------------------------" << std::endl;
		//	- Spline.
		//		Basis spline (B-spline).
		//		Non-uniform rational B-spline (NURBS).
		//		Thin-plate spline (TPS). (?)
		//	- Bezier.
		retval = tinyspline_main(argc, argv);
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
