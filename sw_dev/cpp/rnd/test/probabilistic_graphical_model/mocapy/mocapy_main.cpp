//#include "stdafx.h"
#include <framework/mocapyexceptions.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace mocapy {

}  // namespace mocapy

int mocapy_main(int argc, char *argv[])
{
	void mocapy_discrete_hmm();

	try
	{
		//mocapy_discrete_hmm();  // run-time error
	}
	catch (const mocapy::MocapyExceptions &e)
	{
		std::cout << "mocapy::MocapyExceptions occurred !!!: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

