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

void discrete_hmm();

}  // namespace discrete_hmm

int mocapy_main(int argc, char *argv[])
{
	try
	{
#if defined(WIN32) || defined(_WIN32)
        throw std::runtime_error("not yet supported in Windows");
#else
		my_mocapy::discrete_hmm();
#endif
	}
	catch (const mocapy::MocapyExceptions &e)
	{
		std::cout << "mocapy::MocapyExceptions caught: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}

