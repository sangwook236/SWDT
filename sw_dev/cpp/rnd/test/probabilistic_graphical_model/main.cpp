//#include "stdafx.h"
#if defined(WIN32)
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
	throw std::runtime_error("not yet implemented");
}

void inference_using_belief_propagation()
{
	throw std::runtime_error("not yet implemented");
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

	int pnl_main(int argc, char *argv[]);
	int mocapy_main(int argc, char *argv[]);
	int libdai_main(int argc, char *argv[]);
	int opengm_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		{
            std::cout << "belief propagation (BP) algorithm -----------------------------------" << std::endl;
			//retval = local::inference_using_belief_propagation();  // not yet implemented
			//retval = bp_vision_main(argc, argv);

            std::cout << "\ngraph-cuts algorithm ------------------------------------------------" << std::endl;
			//retval = local::inference_using_graphcut();  // not yet implemented
			//retval = cuda_cut_main(argc, argv);  // not yet implemented

            std::cout << "\nTree-Reweighted (TRW-S) message passing algorithm -------------------" << std::endl;
			//retval = trws_main(argc, argv);  // not yet implemented

            std::cout << "\nquadratic pseudo-boolean optimization (QPBO) algorithm --------------" << std::endl;
			//retval = qpbo_main(argc, argv);  // not yet implemented
		}

		std::cout << "\nhidden Markov model (HMM) -------------------------------------------" << std::endl;
		//retval = hmm_main(argc, argv);

		std::cout << "\nMarkov random field (MRF) -------------------------------------------" << std::endl;
		//retval = middlebury_main(argc, argv);

		std::cout << "\nconditional random field (CRF) --------------------------------------" << std::endl;
		//retval = crfpp_main(argc, argv);
		//retval = hcrf_main(argc, argv);

		std::cout << "\ndynamic Bayesian network (DBN) --------------------------------------" << std::endl;

		std::cout << "\nProbabilistic Networks Library (PNL) --------------------------------" << std::endl;
		//retval = pnl_main(argc, argv);

		std::cout << "\nMocapy++ library ----------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = mocapy_main(argc, argv);
#endif

		std::cout << "\nlibDAI library ---------------------------------" << std::endl;
		//retval = libdai_main(argc, argv);

		std::cout << "\nOpenGM library ------------------------------------------------------" << std::endl;
		// OpenGM library ------------------------------------------
		retval = opengm_main(argc, argv);
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
