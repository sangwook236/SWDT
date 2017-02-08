#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char* argv[])
{
	int openblas_main(int argc, char* argv[]);
	int lapack_main(int argc, char* argv[]);
	int atlas_main(int argc, char* argv[]);
	int eigen_main(int argc, char* argv[]);
	int armadillo_main(int argc, char* argv[]);
	int newmat_main(int argc, char* argv[]);
	int cvm_main(int argc, char* argv[]);
	int mtl_main(int argc, char* argv[]);
	int suitesparse_main(int argc, char* argv[]);
	int superlu_main(int argc, char* argv[]);
	int viennacl_main(int argc, char* argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "OpenBLAS library ----------------------------------------------------" << std::endl;
		//retval = openblas_main(argc, argv);
		std::cout << "\nLinear Algebra PACKage (LAPACK) -------------------------------------" << std::endl;
		//retval = lapack_main(argc, argv);
		std::cout << "\nAutomatically Tuned Linear Algebra Software (ATLAS) -----------------" << std::endl;
		//retval = atlas_main(argc, argv);

		std::cout << "Boost uBLAS library -------------------------------------------------" << std::endl;
		// REF [library] >> Boost library.

		std::cout << "\nEigen library -------------------------------------------------------" << std::endl;
		//	- External library support.
		//		SuiteSparse, Cholmod, UmfPack, SuperLU, Pardiso, PaStiX, SPQR, Metis.
		//	- Unsupported.
		//		Matrix function.
		//		Differentiation.
		//			Numerical differentiation.
		//			Automatic differentiation.
		//				ADOL-C.
		//		Polynomial.
		//		Geometry.
		//		Fast Fourier transform (FFT).
		//		Non-linear optimization.
		//		Spline and spline fitting.
		//			Basis spline (B-spline).
		//		MPFR support.
		//retval = eigen_main(argc, argv);
		std::cout << "\nArmadillo library ---------------------------------------------------" << std::endl;
		//retval = armadillo_main(argc, argv);

		std::cout << "\nNewmat C++ matrix library -------------------------------------------" << std::endl;
		//retval = newmat_main(argc, argv);  // Not yet implemented.
		std::cout << "\nCVM Class Library ---------------------------------------------------" << std::endl;
		//retval = cvm_main(argc, argv);
		std::cout << "\nMatrix Template Library (MTL) ---------------------------------------" << std::endl;
		//retval = mtl_main(argc, argv);

		std::cout << "\nSuiteSparse library -------------------------------------------------" << std::endl;
		//retval = suitesparse_main(argc, argv);  // Not yet implemented.
		std::cout << "\nSuperLU library -----------------------------------------------------" << std::endl;
		//retval = superlu_main(argc, argv);

		std::cout << "\nThe Vienna Computing Library (ViennaCL) -----------------------------" << std::endl;
		//	- Incomplete LU factorization (ILU).
		//	- Multigrid method.
		//		Algebraic multigrid (AMG) method.
		//	- Eigenproblem.
		//		Bisection.
		//		Iterative solver.
		//			Power iteration.
		//			Lanczos algorithm.
		//	- Interfaceing with Boost uBLAS, Eigen, MTL4, Python, and Matlab.
		//	- Nonnegative matrix factorization (NMF).
		//	- Fast Fourier transform (FFT).
		retval = viennacl_main(argc, argv);
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
