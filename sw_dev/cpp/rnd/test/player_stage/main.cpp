//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <libplayerc++/playererror.h>
#include <iostream>


int main(int argc, char *argv[])
{
	void simple_example();
	void rubbish_collecting_robot(int argc, char *argv[]);

	try
	{
		//simple_example();
		rubbish_collecting_robot(argc, argv);
	}
	catch (const PlayerCc::PlayerError &e)
	{
		std::cout << "PlayerCc::PlayerError caught: " << e.GetErrorStr() << std::endl;
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

