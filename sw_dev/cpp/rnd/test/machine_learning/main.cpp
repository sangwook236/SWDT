//include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <ctime>


int main(int argc, char *argv[])
{
	int clustering_main(int argc, char *argv[]);
	int vlfeat_main(int argc, char *argv[]);
	int rl_glue_main(int argc, char *argv[]);

	try
	{
		std::srand((unsigned int)time(NULL));

		// clustering
		clustering_main(argc, argv);
		vlfeat_main(argc, argv);

		// reinforcement learning
		rl_glue_main(argc, argv);
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
