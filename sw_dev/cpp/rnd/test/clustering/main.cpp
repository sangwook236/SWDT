//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>


int main(int argc, char **argv)
{
	void k_means();
	void k_medoids();

	try
	{
		std::srand((unsigned int)time(NULL));

		k_means();  // not yet implemented
		k_medoids();  // not yet implemented
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
