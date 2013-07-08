//#include "stdafx.h"
#include <framework/mocapyexceptions.h>
#include <iostream>
#if defined(WIN32) || defined(_WIN32)
#include <stdexcept>
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mocapy {

void simple_hmm();
void hmm_with_discrete();
void hmm_with_discrete_and_prior();
void hmm_with_gaussian_1d();
void hmm_with_gaussian_2d();
void hmm_with_von_mises_1d();
void hmm_with_von_mises_2d();
void hmm_with_kent();
void factorial_hmm();
void mixture_model_inference();
void hmm_inference();

}  // namespace discrete_hmm

int mocapy_main(int argc, char *argv[])
{
	try
	{
		//my_mocapy::simple_hmm();

		my_mocapy::hmm_with_discrete();
		//my_mocapy::hmm_with_discrete_and_prior();
		//my_mocapy::hmm_with_gaussian_1d();
		//my_mocapy::hmm_with_gaussian_2d();
		//my_mocapy::hmm_with_von_mises_1d();
		//my_mocapy::hmm_with_von_mises_2d();
		//my_mocapy::hmm_with_kent();

		//my_mocapy::factorial_hmm();

		//my_mocapy::mixture_model_inference();
		//my_mocapy::hmm_inference();
	}
	catch (const mocapy::MocapyExceptions &e)
	{
		std::cout << "mocapy::MocapyExceptions caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

