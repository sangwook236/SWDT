//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
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
		retval = gsl_main(argc, argv);
		std::cout << "\nGSLwrap library -----------------------------------------------------" << std::endl;
		//retval = gslwrap_main(argc, argv);

		std::cout << "\nALGLIB library ------------------------------------------------------" << std::endl;
		//	- Linear algebra.
		//	- Interpolation.
		//		Polynomial interpolation.
		//		Rational interpolation.
		//		Spline: linear, Hermite, Catmull-Rom, cubic, and Akima splines.
		//		Bilinear and bicubic spline interpolation.
		//		Fast RBF interpolation/fitting.
		//		Least squares fitting (linear/nonlinear).
		//	- Differentiataion & integration.
		//	- Optimization.
		//	- Statistics.
		//		Distribution.
		//			F-distribution, chi-square distribution, binomial distribution, Poisson distribution, Student's t-distribution, normal distribution, error function.
		//		Hypothesis testing.
		//			Student's t-tests, F-test and chi-square test.
		//			Sign test, Wilcoxon signed-rank test, Mann-Whitney U-test.
		//			Jarque-Bera test, significance test for correlation coefficient.
		//	- Data analysis.
		//		Linear regression, multinomial logit regression.
		//		Neural networks, neural network ensembles.
		//		(Random) decision forest.
		//		Principal component analysis (PCA), linear discriminant analysis (LDA).
		//		Clustering: hierarchical, k-means++.
		//	- Special function.
		//		Orthogonal polynomial.
		//			Chebyshev polynomial, Hermite polynomial, Laguerre polynomial, Legendre polynomial.
		//		(Incomplete) gamma function, psi function.
		//		(Incomplete) beta function.
		//		Bessel functions of integer order.
		//		Elliptic integrals of the first and second kind, Jacobian elliptic functions.
		//		Dawson integral, exponential integral, trigonometric integral, Fresnel integral, Airy function.
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
