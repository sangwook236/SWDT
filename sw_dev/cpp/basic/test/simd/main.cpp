#include "stdafx.h"
#include <iostream>


int main(int argc, char **argv)
{
	void sse();

	try
	{
		sse();
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught !!!" << std::endl;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}

