//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int mfa_main(int argc, char *argv[]);
	int tapkee_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "mixtures of factor analyzers (MFA) ----------------------------------" << std::endl;
		//	-. EM algorithm for mixtures of factor analyzers (MFA).
		//	-. Train three specializations of MFA:
		//		A (single) factor analyzer (FA).
		//		A (single) probabilistic PCA model (PPCA).
		//		A mixture of probabilistic PCA models (MPPCA). (X)
		retval = mfa_main(argc, argv);

		std::cout << "\ntapkee library ------------------------------------------------------" << std::endl;
		//	-. Locally Linear Embedding (LLE) and Kernel Locally Linear Embedding (KLLE).
		//	-. Neighborhood Preserving Embedding (NPE).
		//	-. Local Tangent Space Alignment (LTSA).
		//	-. Linear Local Tangent Space Alignment (LLTSA).
		//	-. Hessian Locally Linear Embedding (HLLE).
		//	-. Laplacian eigenmaps.
		//	-. Locality Preserving Projections (LPP).
		//	-. Diffusion map.
		//	-. Isomap and landmark Isomap.
		//	-. Multidimensional scaling (MDS) and landmark Multidimensional scaling (lMDS).
		//	-. Stochastic Proximity Embedding (SPE).
		//	-. PCA and randomized PCA.
		//	-. Kernel PCA (kPCA).
		//	-. Random projection (RP).
		//	-. Factor analysis (FA).
		//	-. t-Distributed Stochastic Neighbor Embedding (t-SNE).
		//	-. Barnes-Hut-SNE.
		//retval = tapkee_main(argc, argv);

		// REF [file] >> Waffles library in machine learning project.
		//	-. dimensionality reduction, manifold learning, attribute selection, and tools related to NLDR.
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
