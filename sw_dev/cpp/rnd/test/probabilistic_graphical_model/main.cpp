//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char **argv)
{
	void mrf();
	void hmm_forward_backward();
	void hmm_viterbi();
	void hmm_learning();
	void hmm_sample();
	void mocapy_main();

	try
	{
		// Markov network
		//mrf();  // not yet implemented

        // hidden Markov model (HMM)
		//hmm_forward_backward();
		//hmm_viterbi();
		hmm_learning();
		//hmm_sample();

		// dynamic Bayesian network (DBN)

		// Mocapy++ library
		//mocapy_main();
	}
	catch (const std::exception &e)
	{
		std::cout << "exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred !!!" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}

