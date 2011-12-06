#include "stdafx.h"
#include <libplayerc++/playerc++.h>
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
		std::cout << e.GetErrorStr() << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return 0;
}

