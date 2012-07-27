#include "stdafx.h"
#include <iostream>


int main(int argc, char **argv)
{
	void flood_library();  // flood library

	try
	{
		flood_library();  // not yet implemented
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

