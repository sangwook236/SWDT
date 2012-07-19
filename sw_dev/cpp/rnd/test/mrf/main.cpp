//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>


int main(int argc, char **argv)
{
	void middlebury_mrf();
	void bp_vision();
	void cuda_cut();

	try
	{
		std::srand((unsigned int)std::time(NULL));

		middlebury_mrf();

		//bp_vision();
		//cuda_cut();
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc occurred !!!: " << e.what() << std::endl;
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
