#include "stdafx.h"
#include <iostream>


int main(int argc, char **argv)
{
	void basic_function();
	void enumeration_process();

	void hand_gesture();
	void skeleton();

	try
	{
		basic_function();
		//enumeration_process();

		//hand_gesture();
		//skeleton();
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
