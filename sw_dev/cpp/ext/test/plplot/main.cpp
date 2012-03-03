#include "stdafx.h"
#include <iostream>


int main(const int argc, const char * argv[])
{
	void example_x01(int argc, const char **argv);
	void example_x21(int argc, const char **argv);

	try
	{
		// examples
		example_x01(argc, argv);
		//example_x21(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

	return 0;
}

