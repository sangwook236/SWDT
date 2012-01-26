#include "stdafx.h"
#include "viterbi.hpp"
#include <iostream>


#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
	void viterbi_algorithm();

	try
	{
		//
		std::cout << "********** method 1" << std::endl;
		viterbi_algorithm();

		//
		std::cout << "\n********** method 2" << std::endl;
		Viterbi::HMM hmmObj;
		hmmObj.init();
		std::cout << hmmObj;

		Viterbi::forward_viterbi(hmmObj.get_observations(), hmmObj.get_states(), hmmObj.get_start_probability(), hmmObj.get_transition_probability(), hmmObj.get_emission_probability());
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

