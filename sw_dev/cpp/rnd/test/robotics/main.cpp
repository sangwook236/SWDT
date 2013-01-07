//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	int mrpt_main(int argc, char *argv[]);
	int player_stage_main(int argc, char *argv[]);

	try
	{
		//mrpt_main(argc, argv);  // compile-time error
		player_stage_main(argc, argv);
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		return -1;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		return -1;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}
