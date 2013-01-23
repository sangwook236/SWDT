//#include "stdafx.h"
#include <framework/mocapyexceptions.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mocapy {

void mocapy_discrete_hmm();

}  // namespace my_mocapy

int mocapy_main(int argc, char *argv[])
{
	try
	{
		//my_mocapy::mocapy_discrete_hmm();  // compile-time error
	}
	catch (const mocapy::MocapyExceptions &e)
	{
		std::cout << "mocapy::MocapyExceptions caught: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

