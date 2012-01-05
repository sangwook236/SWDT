#include "stdafx.h"
#include <iostream>
#include <ctime>


#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
	void bayesian_network();
	void bnet();
	void mnet();
	void mrf();
	void dbn();
	void hmm();

	try
	{
		std::srand((unsigned int)std::time(NULL));

		// Bayesian network
		//bayesian_network();
		//bnet();

		// Markov network
		//mnet();
		//mrf();

		// dynamic Bayesian network
		//dbn();
		hmm();
	}
	catch (const std::exception &e)
	{
		std::wcout << L"exception occurred !!!: " << e.what() << std::endl;
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

