//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>


int main(int argc, char *argv[])
{
	int ransac_main(int argc, char *argv[]);

	try
	{
		std::srand((unsigned int)time(NULL));

		ransac_main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred !!!" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
