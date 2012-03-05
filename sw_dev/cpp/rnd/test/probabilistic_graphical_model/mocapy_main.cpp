//#include "stdafx.h"
#include <framework/MocapyExceptions.h>
#include <iostream>


void mocapy_main()
{
	void mocapy_discrete_hmm();

	try
	{
		mocapy_discrete_hmm();
	}
	catch (const mocapy::MocapyExceptions &e)
	{
		std::cout << "mocapy::MocapyExceptions occurred !!!: " << e.what() << std::endl;
	}
}

