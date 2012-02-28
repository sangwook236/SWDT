#include "stdafx.h"
#include <iostream>


int main(const int argc, const char * argv[])
{
	void x01_main(int argc, const char **argv);
	void x21_main(int argc, const char **argv);

	try
	{
		// examples
		x01_main(argc, argv);
		//x21_main(argc, argv);
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

