//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


namespace {
namespace local {

void inference_using_graphcut()
{
	throw std::runtime_error("Not yet implemented");
}

void inference_using_belief_propagation()
{
	throw std::runtime_error("Not yet implemented");
}

}  // namespace local
}  // unnamed namespace

int main(int argc, char *argv[])
{
	int bp_vision_main(int argc, char *argv[]);
	int cuda_cut_main(int argc, char *argv[]);
	int trws_main(int argc, char *argv[]);
	int qpbo_main(int argc, char *argv[]);

	int hmm_main(int argc, char *argv[]);

	int middlebury_main(int argc, char *argv[]);

	int crfpp_main(int argc, char *argv[]);
	int hcrf_main(int argc, char *argv[]);
	int densecrf_main(int argc, char *argv[]);

	int pnl_main(int argc, char *argv[]);
	int mocapy_main(int argc, char *argv[]);
	int libdai_main(int argc, char *argv[]);
	int opengm_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		{
            std::cout << "Belief propagation (BP) algorithm -----------------------------------" << std::endl;
			//retval = local::inference_using_belief_propagation();  // Not yet implemented.
			//retval = bp_vision_main(argc, argv);

            std::cout << "\nGraph-cuts algorithm ------------------------------------------------" << std::endl;
			//retval = local::inference_using_graphcut();  // Not yet implemented.
			//retval = cuda_cut_main(argc, argv);  // Not yet implemented.

            std::cout << "\nTree-Reweighted (TRW-S) message passing algorithm -------------------" << std::endl;
			//retval = trws_main(argc, argv);  // Not yet implemented.

            std::cout << "\nQuadratic pseudo-boolean optimization (QPBO) algorithm --------------" << std::endl;
			//retval = qpbo_main(argc, argv);  // Not yet implemented.
		}

		std::cout << "\nHidden Markov model (HMM) -------------------------------------------" << std::endl;
		//retval = hmm_main(argc, argv);

		std::cout << "\nMarkov random field (MRF) -------------------------------------------" << std::endl;
		//retval = middlebury_main(argc, argv);

		std::cout << "\nConditional random field (CRF) --------------------------------------" << std::endl;
		//retval = crfpp_main(argc, argv);
		//	- Hidden CRF (HCRF).
		//	- Laten-dynamc CRF (LDCRF).
		//retval = hcrf_main(argc, argv);
		//	- Fully-connected (dense) CRF.
		retval = densecrf_main(argc, argv);

		std::cout << "\nDynamic Bayesian network (DBN) --------------------------------------" << std::endl;

		std::cout << "\nProbabilistic Networks Library (PNL) --------------------------------" << std::endl;
		//retval = pnl_main(argc, argv);

		std::cout << "\nMocapy++ library ----------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = mocapy_main(argc, argv);
#else
        std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nlibDAI library ------------------------------------------------------" << std::endl;
		//	- Exact inference by brute force enumeration.
		//	- Exact inference by junction-tree methods.
		//	- Mean field.
		//	- Loopy belief propagation (LBP).
		//	- Fractional belief propagation.
		//	- Tree-reweighted belief propagation (TRBP).
		//	- Tree expectation propagation.
		//	- Generalized belief propagation (GBP).
		//	- Double-loop GBP.
		//	- Various variants of loop corrected belief propagation.
		//	- Gibbs sampler.
		//	- Conditioned belief propagation.
		//	- Decimation algorithm.
		//retval = libdai_main(argc, argv);

		std::cout << "\nOpenGM library ------------------------------------------------------" << std::endl;
		//	- Combinatorial/gobal optimal method.
		//	- Linear programming relaxation.
		//	- Message passing method.
		//	- Move making method.
		//	- Sampling.
		//	- Wrapped external code for discrete graphical models.
		//retval = opengm_main(argc, argv);
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
