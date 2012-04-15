//#include "stdafx.h"
#include <pnl_dll.hpp>
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>


#if defined(GetMessage)
#undef GetMessage
#endif

int main(int argc, char **argv)
{
	void bayesian_network_example();
	void bayesian_network();
	void mnet_example();
	void mrf_example();
	void mrf2();
	void dbn_example();
	void dbn();
	void hmm();
	void viterbi_segmentation();
	void super_resolution();  // not yet implemented

	try
	{
		std::srand((unsigned int)std::time(NULL));

		std::ofstream logFileStream("pnl.log", std::ios::app);
		pnl::LogDrvStream logDriver(&logFileStream, pnl::eLOG_ALL, pnl::eLOGSRV_ALL);

		// Bayesian network
		//bayesian_network_example();  // from an example in "PNL: User Guide & Reference Manual"
		//bayesian_network();

		// Markov network
		//mnet_example();  // from a test of PNL
		//mrf_example();  // from a test of PNL
		//mrf2();

		// dynamic Bayesian network
		//dbn_example();  // from an example in "PNL: User Guide & Reference Manual"
		//dbn();
		hmm();

		// application
		//viterbi_segmentation();
		//super_resolution();  // not yet implemented
	}
	catch (const pnl::CException &e)
	{
		std::cout << "OpenPNL exception occurred !!!: " << e.GetMessage() << std::endl;
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

