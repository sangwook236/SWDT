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

	int hmm_main(int argc, char *argv[]);

	int middlebury_main(int argc, char *argv[]);

	int crfpp_main(int argc, char *argv[]);
	int hcrf_main(int argc, char *argv[]);

	int pnl_main(int argc, char *argv[]);
	int mocapy_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));

		{
			//retval = local::inference_using_graphcut();  // not yet implemented
			//retval = local::inference_using_belief_propagation();  // not yet implemented

			// belief propagation (BP) algorithm -----------------------
			//retval = bp_vision_main(argc, argv);

			// graph-cuts algorithm ------------------------------------
			//retval = cuda_cut_main(argc, argv);  // not yet implemented
		}

        // hidden Markov model (HMM) ----------------------
		//retval = hmm_main(argc, argv);

		// Markov random field (MRF) ----------------------
		//retval = middlebury_main(argc, argv);

		// conditional random field (CRF) -----------------
		//retval = crfpp_main(argc, argv);
		retval = hcrf_main(argc, argv);

		// dynamic Bayesian network (DBN) -----------------

		// PNL library ------------------------------------
		//retval = pnl_main(argc, argv);

		// Mocapy++ library -------------------------------
		//retval = mocapy_main(argc, argv);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc occurred: " << e.what() << std::endl;
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
