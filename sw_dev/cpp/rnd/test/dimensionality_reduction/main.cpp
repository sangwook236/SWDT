//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	int tapkee_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "tapkee library ------------------------------------------------------" << std::endl;
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
		retval = tapkee_main(argc, argv);

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
