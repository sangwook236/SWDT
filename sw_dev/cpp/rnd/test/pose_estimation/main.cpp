//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int parts_based_detector_main(int argc, char *argv[]);

	try
	{
		parts_based_detector_main(argc, argv);  // not yet implemented
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}

