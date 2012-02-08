#include "stdafx.h"
#include <pnl_dll.hpp>
#include <vld/vld.h>
#include <iostream>
#include <ctime>


#if defined(GetMessage)
#undef GetMessage
#endif

#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
	void bayesian_network_example();
	void bayesian_network();
	void mnet_example();
	void mrf_example();
	void dbn_example();
	void dbn();
	void hmm();
	void viterbi_segmentation();
	void super_resolution();  // not yet implemented

	try
	{
		std::srand((unsigned int)std::time(NULL));

		// Bayesian network
		//bayesian_network_example();  // from an example in "PNL: User Guide & Reference Manual"
		bayesian_network();

		// Markov network
		//mnet_example();  // from a test of PNL
		//mrf_example();  // from a test of PNL

		// dynamic Bayesian network
		//dbn_example();  // from an example in "PNL: User Guide & Reference Manual"
		//dbn();
		//hmm();

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
		std::wcout << L"unknown exception occurred !!!" << std::endl;
	}

	std::wcout << L"press any key to exit ..." << std::endl;
	std::wcout.flush();
	std::wcin.get();

	return 0;
}

