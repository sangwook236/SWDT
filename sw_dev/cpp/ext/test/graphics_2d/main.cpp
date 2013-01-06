//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int cairo_main(int argc, char *argv[]);
	int gd_main(int argc, char *argv[]);
	int devil_main(int argc, char *argv[]);

	try
	{
		cairo_main(argc, argv);
		//gd_main(argc, argv);
		
		//devil_main(argc, argv);
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
