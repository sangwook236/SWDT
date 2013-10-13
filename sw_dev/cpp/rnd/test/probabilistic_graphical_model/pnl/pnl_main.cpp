//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <iostream>


#if defined(GetMessage)
#undef GetMessage
#endif

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_pnl {

void bayesian_network_example();
void bayesian_network();
void mnet_example();
void mrf_example();
void mrf2();
void dbn_example();
void dbn();
void hmm();
void viterbi_segmentation();
void super_resolution();  // not yet implemented.

}  // namespace my_pnl

int pnl_main(int argc, char *argv[])
{
	try
	{
		std::ofstream logFileStream("pnl.log", std::ios::app);
		pnl::LogDrvStream logDriver(&logFileStream, pnl::eLOG_ALL, pnl::eLOGSRV_ALL);

		// Bayesian network
		//my_pnl::bayesian_network_example();  // from an example in "PNL: User Guide & Reference Manual".
		//my_pnl::bayesian_network();

		// Markov network
		//my_pnl::mnet_example();  // from a test of PNL.
		//my_pnl::mrf_example();  // from a test of PNL.
		//my_pnl::mrf2();

		// dynamic Bayesian network
		//my_pnl::dbn_example();  // from an example in "PNL: User Guide & Reference Manual".
		//my_pnl::dbn();
		//my_pnl::hmm();

		// application
		my_pnl::viterbi_segmentation();
		//my_pnl::super_resolution();  // not yet implemented.
	}
	catch (const pnl::CException &e)
	{
		std::cout << "OpenPNL exception caught: " << e.GetMessage() << std::endl;
		return -1;
	}

	return 0;
}
